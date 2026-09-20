use aien_inference_abi::{FinishReason, SamplingParams, SequenceRequest};
use aien_platform::{InferenceWork, KvHandle, ModelHandle, Priority, SequenceId, Ticks};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Immutable, reference-counted prompt token buffer.
/// Guarantees zero allocation overhead across sequence forks and subagent branches.
pub type PromptHandle = Arc<[u32]>;

/// Opaque identifier for a decoupled completion sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompletionSinkId(pub u64);

/// Structured lifecycle event emitted for a sequence during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionEvent {
    Token {
        seq_id: SequenceId,
        token: u32,
    },
    Finished {
        seq_id: SequenceId,
        finish_reason: FinishReason,
        total_tokens: usize,
    },
    Error {
        seq_id: SequenceId,
        message: String,
    },
}

/// Decoupled event consumer trait free from sockets, threads, and processes.
pub trait CompletionSink: Send + Sync {
    fn emit(&self, event: CompletionEvent);
}

/// Channel-based completion sink implementation for async task integration.
pub struct ChannelCompletionSink {
    sender: tokio::sync::mpsc::UnboundedSender<CompletionEvent>,
}

impl ChannelCompletionSink {
    pub fn new(sender: tokio::sync::mpsc::UnboundedSender<CompletionEvent>) -> Self {
        Self { sender }
    }

    pub fn channel() -> (Self, tokio::sync::mpsc::UnboundedReceiver<CompletionEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (Self::new(tx), rx)
    }
}

impl CompletionSink for ChannelCompletionSink {
    fn emit(&self, event: CompletionEvent) {
        let _ = self.sender.send(event);
    }
}

/// Event router managing active completion sinks.
#[derive(Default)]
pub struct CompletionRouter {
    sinks: HashMap<CompletionSinkId, Arc<dyn CompletionSink>>,
    next_id: u64,
}

impl CompletionRouter {
    pub fn new() -> Self {
        Self {
            sinks: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn register(&mut self, sink: Arc<dyn CompletionSink>) -> CompletionSinkId {
        let id = CompletionSinkId(self.next_id);
        self.next_id += 1;
        self.sinks.insert(id, sink);
        id
    }

    pub fn unregister(&mut self, id: CompletionSinkId) -> Option<Arc<dyn CompletionSink>> {
        self.sinks.remove(&id)
    }

    pub fn emit(&self, sink_id: CompletionSinkId, event: CompletionEvent) {
        if let Some(sink) = self.sinks.get(&sink_id) {
            sink.emit(event);
        }
    }

    pub fn broadcast(&self, event: CompletionEvent) {
        for sink in self.sinks.values() {
            sink.emit(event.clone());
        }
    }
}

/// Span of prompt tokens scheduled for prefill execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillSpan {
    pub seq_id: SequenceId,
    pub start_pos: usize,
    pub length: usize,
}

/// Single autoregressive decode step scheduled for an active sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeItem {
    pub seq_id: SequenceId,
    pub token_pos: usize,
}

/// Structured hardware execution plan dividing work into prefill spans and decode steps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchPlan {
    pub step_id: u64,
    pub prefill_spans: Vec<PrefillSpan>,
    pub decode_items: Vec<DecodeItem>,
    pub total_tokens: usize,
}

/// Explicit lifecycle phases for a sequence record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequencePhase {
    Waiting,
    Prefill,
    PrefillInFlight,
    Decode,
    DecodeInFlight,
    Preempted,
    Finished,
    Cancelled,
}

/// Authoritative runtime state record for an individual sequence.
#[derive(Debug, Clone)]
pub struct SequenceRecord {
    pub seq_id: SequenceId,
    pub prompt: PromptHandle,
    pub model: ModelHandle,
    pub kv: KvHandle,
    pub priority: Priority,
    pub deadline: Option<Ticks>,
    pub branch_parent: Option<SequenceId>,
    pub next_token_budget: u32,
    pub phase: SequencePhase,
    pub tokens_generated: usize,
    pub prompt_tokens_prefilled: usize,
    pub is_prefilled: bool,
    pub sink_id: Option<CompletionSinkId>,
    pub sampling_params: SamplingParams,
    pub arrival_time_ns: u64,
    pub generated_tokens: Vec<u32>,
}

impl SequenceRecord {
    pub fn new(
        seq_id: SequenceId,
        prompt: PromptHandle,
        model: ModelHandle,
        kv: KvHandle,
        priority: Priority,
        deadline: Option<Ticks>,
        branch_parent: Option<SequenceId>,
        next_token_budget: u32,
        sampling_params: SamplingParams,
        sink_id: Option<CompletionSinkId>,
    ) -> Self {
        Self {
            seq_id,
            prompt,
            model,
            kv,
            priority,
            deadline,
            branch_parent,
            next_token_budget,
            phase: SequencePhase::Waiting,
            tokens_generated: 0,
            prompt_tokens_prefilled: 0,
            is_prefilled: false,
            sink_id,
            sampling_params,
            arrival_time_ns: 0,
            generated_tokens: Vec::new(),
        }
    }

    pub fn from_work(
        work: InferenceWork,
        prompt: PromptHandle,
        sampling_params: SamplingParams,
        sink_id: Option<CompletionSinkId>,
    ) -> Self {
        Self {
            seq_id: work.sequence,
            prompt,
            model: work.model,
            kv: work.kv,
            priority: work.priority,
            deadline: work.deadline,
            branch_parent: work.branch_parent,
            next_token_budget: work.next_token_budget,
            phase: SequencePhase::Waiting,
            tokens_generated: 0,
            prompt_tokens_prefilled: 0,
            is_prefilled: false,
            sink_id,
            sampling_params,
            arrival_time_ns: 0,
            generated_tokens: Vec::new(),
        }
    }

    pub fn from_request(request: SequenceRequest, sink_id: Option<CompletionSinkId>) -> Self {
        let prompt_slice: Box<[u32]> = request.prompt_tokens.into_boxed_slice();
        let prompt_handle: PromptHandle = Arc::from(prompt_slice);
        let priority = match request.priority {
            0 => Priority::Background,
            1 => Priority::Normal,
            2 => Priority::Interactive,
            _ => Priority::Realtime,
        };

        Self {
            seq_id: request.request_id,
            prompt: prompt_handle,
            model: ModelHandle(0),
            kv: KvHandle(request.request_id),
            priority,
            deadline: None,
            branch_parent: None,
            next_token_budget: request.sampling_params.max_tokens as u32,
            phase: SequencePhase::Waiting,
            tokens_generated: 0,
            prompt_tokens_prefilled: 0,
            is_prefilled: false,
            sink_id,
            sampling_params: request.sampling_params,
            arrival_time_ns: request.arrival_time_ns,
            generated_tokens: Vec::new(),
        }
    }

    pub fn remaining_prefill_tokens(&self) -> usize {
        self.prompt
            .len()
            .saturating_sub(self.prompt_tokens_prefilled)
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt.len() + self.tokens_generated
    }

    pub fn is_finished(&self) -> bool {
        self.phase == SequencePhase::Finished
            || self.tokens_generated >= self.sampling_params.max_tokens
            || (self.next_token_budget > 0
                && self.tokens_generated >= self.next_token_budget as usize)
    }

    pub fn to_request(&self) -> SequenceRequest {
        SequenceRequest {
            request_id: self.seq_id,
            prompt_tokens: self.prompt.to_vec(),
            sampling_params: self.sampling_params.clone(),
            arrival_time_ns: self.arrival_time_ns,
            priority: self.priority as u8,
        }
    }
}

/// Central state arena managing active sequence records with zero-copy branching.
#[derive(Default)]
pub struct SequenceArena {
    records: HashMap<SequenceId, SequenceRecord>,
    total_tokens_generated: usize,
    total_prefilled_tokens: usize,
}

impl SequenceArena {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            total_tokens_generated: 0,
            total_prefilled_tokens: 0,
        }
    }

    pub fn insert(&mut self, record: SequenceRecord) -> Result<(), String> {
        let seq_id = record.seq_id;
        if self.records.contains_key(&seq_id) {
            return Err(format!("Sequence {} already exists in arena", seq_id));
        }
        self.records.insert(seq_id, record);
        Ok(())
    }

    pub fn get(&self, seq_id: &SequenceId) -> Option<&SequenceRecord> {
        self.records.get(seq_id)
    }

    pub fn get_mut(&mut self, seq_id: &SequenceId) -> Option<&mut SequenceRecord> {
        self.records.get_mut(seq_id)
    }

    pub fn remove(&mut self, seq_id: &SequenceId) -> Option<SequenceRecord> {
        self.records.remove(seq_id)
    }

    pub fn contains(&self, seq_id: SequenceId) -> bool {
        self.records.contains_key(&seq_id)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Zero-copy sequence fork preserving immutable prompt handle.
    pub fn fork(
        &mut self,
        parent_id: SequenceId,
        child_id: SequenceId,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<SequenceRecord, String> {
        let parent = self
            .records
            .get(&parent_id)
            .ok_or_else(|| format!("Parent sequence {} not found in arena", parent_id))?;

        if self.records.contains_key(&child_id) {
            return Err(format!(
                "Child sequence {} already exists in arena",
                child_id
            ));
        }

        let child = SequenceRecord {
            seq_id: child_id,
            prompt: Arc::clone(&parent.prompt),
            model: parent.model,
            kv: KvHandle(child_id),
            priority: parent.priority,
            deadline: parent.deadline,
            branch_parent: Some(parent_id),
            next_token_budget: parent.next_token_budget,
            phase: SequencePhase::Decode,
            tokens_generated: parent.tokens_generated,
            prompt_tokens_prefilled: parent.prompt_tokens_prefilled,
            is_prefilled: parent.is_prefilled,
            sink_id,
            sampling_params: parent.sampling_params.clone(),
            arrival_time_ns: parent.arrival_time_ns,
            generated_tokens: parent.generated_tokens.clone(),
        };

        self.records.insert(child_id, child.clone());
        Ok(child)
    }

    pub fn active_ids(&self) -> Vec<SequenceId> {
        let mut ids: Vec<SequenceId> = self.records.keys().copied().collect();
        ids.sort();
        ids
    }

    pub fn record_generated_token(&mut self, seq_id: SequenceId, token: u32) {
        if let Some(record) = self.records.get_mut(&seq_id) {
            record.tokens_generated += 1;
            record.generated_tokens.push(token);
            self.total_tokens_generated += 1;
        }
    }

    pub fn record_prefilled_tokens(&mut self, seq_id: SequenceId, count: usize) {
        if let Some(record) = self.records.get_mut(&seq_id) {
            record.prompt_tokens_prefilled += count;
            if record.prompt_tokens_prefilled >= record.prompt.len() {
                record.is_prefilled = true;
                record.phase = SequencePhase::Decode;
            } else {
                record.phase = SequencePhase::Prefill;
            }
            self.total_prefilled_tokens += count;
        }
    }

    pub fn total_tokens_generated(&self) -> usize {
        self.total_tokens_generated
    }

    pub fn total_prefilled_tokens(&self) -> usize {
        self.total_prefilled_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_handle_sharing_and_zero_copy_fork() {
        let prompt_tokens = vec![1, 2, 3, 4, 5];
        let handle: PromptHandle = Arc::from(prompt_tokens.into_boxed_slice());

        let mut arena = SequenceArena::new();
        let record = SequenceRecord::new(
            1,
            handle.clone(),
            ModelHandle(0),
            KvHandle(1),
            Priority::Normal,
            None,
            None,
            128,
            SamplingParams::default(),
            None,
        );

        arena.insert(record).unwrap();
        assert_eq!(arena.len(), 1);

        let child = arena.fork(1, 2, None).unwrap();
        assert_eq!(arena.len(), 2);
        assert_eq!(child.seq_id, 2);
        assert_eq!(child.branch_parent, Some(1));
        assert!(Arc::ptr_eq(&child.prompt, &handle));
    }

    #[test]
    fn test_completion_router_registration_and_dispatch() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = Arc::new(ChannelCompletionSink::new(tx));

        let mut router = CompletionRouter::new();
        let sink_id = router.register(sink);

        router.emit(
            sink_id,
            CompletionEvent::Token {
                seq_id: 10,
                token: 42,
            },
        );

        let event = rx.try_recv().unwrap();
        match event {
            CompletionEvent::Token { seq_id, token } => {
                assert_eq!(seq_id, 10);
                assert_eq!(token, 42);
            }
            _ => panic!("Unexpected event"),
        }
    }
}
