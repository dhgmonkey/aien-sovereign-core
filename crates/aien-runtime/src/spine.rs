//! Canonical AienRuntimeSpine
//! Unifies SequenceArena, AienScheduler, AienKvManager, WorldStore, ContextComposer,
//! and SwarmManager in a single process address space.

use crate::context::ContextComposer;
use crate::control::{
    ControlCommand, ControlEnvelope, ControlResponse, RuntimeController, RuntimeStatusReport,
};
use crate::sequence::{SequenceArena, SequenceId, SequenceState};
use crate::swarm::{SwarmConfig, SwarmManager};
use crate::world::WorldStore;

use aien_inference_abi::{AienInferenceBackend, SamplingParams, StepMetrics};
use aien_kv_cache::AienKvManager;
use aien_scheduler::{
    AienScheduler, CompletionSink, CompletionSinkId, PromptHandle, SchedulerConfig,
};
use parking_lot::RwLock;
use std::sync::Arc;

pub struct AienRuntimeSpine {
    pub arena: SequenceArena,
    pub kv_manager: Arc<RwLock<AienKvManager>>,
    pub scheduler: AienScheduler,
    pub world_store: WorldStore,
    pub context_composer: ContextComposer,
    pub swarm_manager: SwarmManager,
    pub controller: RuntimeController,
    pub step_counter: u64,
}

impl AienRuntimeSpine {
    pub fn new(
        arena_capacity: usize,
        scheduler_config: SchedulerConfig,
        kv_manager: Arc<RwLock<AienKvManager>>,
    ) -> Self {
        let scheduler = AienScheduler::new(scheduler_config, kv_manager.clone());
        Self {
            arena: SequenceArena::new(arena_capacity),
            kv_manager,
            scheduler,
            world_store: WorldStore::new(),
            context_composer: ContextComposer::new(),
            swarm_manager: SwarmManager::new(),
            controller: RuntimeController::new(),
            step_counter: 0,
        }
    }

    /// Submits structured InferenceWork with PromptHandle and optional completion sink.
    pub fn submit_inference_work(
        &mut self,
        work: aien_platform::queue::InferenceWork,
        prompt: PromptHandle,
        sampling_params: Option<SamplingParams>,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<(), String> {
        self.scheduler
            .submit_work(work, prompt, sampling_params, sink_id)
    }

    /// Submits a prompt ticket to the scheduler for execution, constructing default InferenceWork.
    pub fn submit_work(
        &mut self,
        prompt: PromptHandle,
        sampling_params: SamplingParams,
        priority: u8,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<u64, String> {
        self.step_counter += 1;
        let seq_id = self.step_counter;
        let plat_priority = match priority {
            0 => aien_platform::Priority::Background,
            1 => aien_platform::Priority::Normal,
            2 => aien_platform::Priority::Interactive,
            _ => aien_platform::Priority::Realtime,
        };
        let work = aien_platform::queue::InferenceWork {
            sequence: seq_id,
            model: aien_platform::ModelHandle(1),
            kv: aien_platform::queue::KvHandle(seq_id),
            priority: plat_priority,
            deadline: None,
            branch_parent: None,
            next_token_budget: sampling_params.max_tokens as u32,
        };
        self.scheduler
            .submit_work(work, prompt, Some(sampling_params), sink_id)?;
        Ok(seq_id)
    }

    /// Registers a completion sink for streaming output events.
    pub fn register_completion_sink(&mut self, sink: Arc<dyn CompletionSink>) -> CompletionSinkId {
        self.scheduler.register_completion_sink(sink)
    }

    /// Zero-copy subagent sequence branching with completion sink routing.
    pub fn fork_subagent(
        &mut self,
        parent_id: u64,
        child_id: u64,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<(), String> {
        self.scheduler
            .fork_subagent_with_sink(parent_id, child_id, sink_id)
    }

    /// Executes runtime steps in a closed loop until all active requests complete or max_steps is reached.
    pub async fn run_until_complete<B: AienInferenceBackend>(
        &mut self,
        backend: &mut B,
        max_steps: usize,
    ) -> Result<Vec<StepMetrics>, String> {
        let mut all_metrics = Vec::new();
        for _ in 0..max_steps {
            if self.scheduler.waiting_count() == 0
                && self.scheduler.running_count() == 0
                && self.scheduler.preempted_count() == 0
            {
                break;
            }
            if let Some(metrics) = self.step(backend).await? {
                all_metrics.push(metrics);
            }
        }
        Ok(all_metrics)
    }

    /// Advances the engine by one transactional step.
    pub async fn step<B: AienInferenceBackend>(
        &mut self,
        backend: &mut B,
    ) -> Result<Option<StepMetrics>, String> {
        self.step_counter += 1;

        // Execute scheduler step via backend
        let step_result = self.scheduler.step(backend).await?;

        if let Some((outputs, metrics)) = step_result {
            // Generational SequenceId validation on step completion
            for output in &outputs {
                match output {
                    aien_inference_abi::DecodeOutput::Token {
                        request_id,
                        token_id: _,
                        logprob: _,
                    } => {
                        let seq_id = SequenceId::from_u64(*request_id);
                        if let Some(record) = self.arena.get_mut(seq_id) {
                            record.generated_tokens += 1;
                        }
                    }
                    aien_inference_abi::DecodeOutput::Finished {
                        request_id,
                        reason: _,
                        total_tokens: _,
                    } => {
                        let seq_id = SequenceId::from_u64(*request_id);
                        if let Some(record) = self.arena.get_mut(seq_id) {
                            record.state = SequenceState::Completed;
                        }
                        self.arena.free(seq_id);
                    }
                }
            }
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }

    /// Launches a swarm of agents sharing an immutable root World and physical KV blocks.
    pub fn launch_swarm(
        &mut self,
        config: SwarmConfig,
        prompt_tokens: &[u32],
    ) -> Result<u64, String> {
        let mut kv = self.kv_manager.write();
        let swarm_id = self.swarm_manager.launch_swarm(
            config,
            &mut self.arena,
            &mut kv,
            &mut self.world_store,
            prompt_tokens,
            self.step_counter,
        )?;

        // Enqueue branch requests into scheduler
        if let Some(swarm) = self.swarm_manager.get_swarm(swarm_id) {
            for &child_seq in &swarm.branch_sequences {
                let req = aien_inference_abi::SequenceRequest {
                    request_id: child_seq.as_u64(),
                    prompt_tokens: prompt_tokens.to_vec(),
                    sampling_params: aien_inference_abi::SamplingParams {
                        temperature: 0.7,
                        top_p: 0.95,
                        max_tokens: swarm.config.max_tokens_per_branch,
                        stop_token_ids: vec![2],
                    },
                    arrival_time_ns: self.step_counter,
                    priority: swarm.config.priority,
                };
                self.scheduler.submit_request(req);
            }
        }

        Ok(swarm_id)
    }

    /// Handles a typed operator command with idempotency verification.
    pub fn handle_control_command(&mut self, envelope: ControlEnvelope) -> ControlResponse {
        if self
            .controller
            .is_operation_processed(envelope.operation_id)
        {
            return ControlResponse::Error(format!(
                "Operation {} already processed",
                envelope.operation_id
            ));
        }

        let resp = match envelope.command {
            ControlCommand::LaunchSwarm(req) => {
                let config = SwarmConfig {
                    model_handle: req.model_handle,
                    branch_count: req.branch_count,
                    max_active_sequences: req.max_active_sequences,
                    max_tokens_per_branch: req.max_tokens_per_branch,
                    root_world_id: req.root_world_id,
                    priority: req.priority,
                };
                match self.launch_swarm(config, &req.prompt_tokens) {
                    Ok(swarm_id) => {
                        self.controller
                            .mark_operation_processed(envelope.operation_id);
                        ControlResponse::SwarmAccepted {
                            swarm_id,
                            operation_id: envelope.operation_id,
                        }
                    }
                    Err(e) => ControlResponse::Error(e),
                }
            }
            ControlCommand::CancelSwarm(swarm_id) => {
                match self.swarm_manager.cancel_swarm(swarm_id, &mut self.arena) {
                    Ok(()) => {
                        self.controller
                            .mark_operation_processed(envelope.operation_id);
                        ControlResponse::SwarmCancelled { swarm_id }
                    }
                    Err(e) => ControlResponse::Error(e),
                }
            }
            ControlCommand::GetRuntimeStatus => ControlResponse::Status(self.status_report()),
            ControlCommand::InspectSwarm(swarm_id) => {
                if let Some(swarm) = self.swarm_manager.get_swarm(swarm_id) {
                    let _ = swarm;
                    ControlResponse::Status(self.status_report())
                } else {
                    ControlResponse::Error(format!("Swarm {} not found", swarm_id))
                }
            }
        };

        resp
    }

    pub fn status_report(&self) -> RuntimeStatusReport {
        let kv = self.kv_manager.read();
        let metrics = kv.metrics();

        RuntimeStatusReport {
            active_sequences: self.arena.active_count(),
            active_swarms: self.swarm_manager.active_swarm_count(),
            active_worlds: self.world_store.active_world_count(),
            free_kv_blocks: kv.free_block_count(),
            total_kv_blocks: kv.free_block_count() + kv.allocated_block_count(),
            shared_kv_pages: metrics.shared_pages,
            cow_faults: metrics.cow_faults,
            gpu_utilization_pct: 0.0,
        }
    }
}
