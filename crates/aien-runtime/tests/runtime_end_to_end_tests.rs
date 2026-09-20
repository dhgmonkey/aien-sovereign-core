//! End-to-end integration tests: Scheduler ticket submission -> Blackwell transformer -> CompletionSink channel.
//! Verifies zero socket, zero HTTP, and zero RPC boundaries in the entire execution path.

use aien_inference_abi::{
    BlackwellGb10Backend, ModelConfig, NativeTransformerBackend, ReferenceCpuBackend,
    SamplingParams, TransformerWeights,
};
use aien_kv_cache::{create_shared_kv_manager, KvDType, KvPoolConfig};
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::{ChannelCompletionSink, CompletionEvent, PromptHandle, SchedulerConfig};
use std::sync::Arc;

fn test_micro_model_config() -> ModelConfig {
    ModelConfig {
        model_id: "aien-micro-v1".to_string(),
        max_sequence_length: 512,
        block_size: 16,
        num_layers: 2,
        num_heads: 4,
        head_dim: 16,
        num_kv_heads: 2,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    }
}

#[tokio::test]
async fn test_end_to_end_ticket_submission_and_channel_streaming_cpu() {
    let config = test_micro_model_config();
    let total_blocks = 64;
    let block_size = 16;

    let pool_cfg = KvPoolConfig {
        num_blocks: total_blocks,
        block_size,
        num_layers: config.num_layers,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        dtype: KvDType::Bf16,
    };

    let kv_manager = create_shared_kv_manager(total_blocks, block_size);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();

    let sched_cfg = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 4,
    };

    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());

    let weights = TransformerWeights::reference_test_weights(&config);
    let mut backend = NativeTransformerBackend::with_shared_kv_and_backend(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        kv_manager,
    );

    // 1. Create in-memory channel sink with zero socket/RPC boundary
    let (sink, mut rx) = ChannelCompletionSink::channel();
    let sink_id = spine.register_completion_sink(Arc::new(sink));

    // 2. Submit immutable PromptHandle ticket
    let prompt: PromptHandle = Arc::from([1u32, 10, 25, 42].as_slice());
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 5,
        stop_token_ids: vec![0],
    };

    let seq_id = spine
        .submit_work(prompt, sampling, 1, Some(sink_id))
        .unwrap();

    // 3. Drive execution steps until completion
    let metrics = spine.run_until_complete(&mut backend, 20).await.unwrap();
    assert!(!metrics.is_empty(), "Must have executed at least 1 step");

    // 4. Collect streamed events from channel
    let mut received_tokens = Vec::new();
    let mut saw_finished = false;

    while let Ok(event) = rx.try_recv() {
        match event {
            CompletionEvent::Token { seq_id: sid, token } => {
                assert_eq!(sid, seq_id);
                received_tokens.push(token);
            }
            CompletionEvent::Finished {
                seq_id: sid,
                total_tokens,
                finish_reason,
            } => {
                assert_eq!(sid, seq_id);
                saw_finished = true;
                eprintln!(
                    "Finished event: total_tokens={}, received_tokens={}, reason={:?}",
                    total_tokens,
                    received_tokens.len(),
                    finish_reason
                );
            }
            CompletionEvent::Error {
                seq_id: sid,
                message,
            } => {
                panic!("Unexpected error for seq_id {}: {}", sid, message);
            }
        }
    }

    assert!(
        saw_finished,
        "Expected CompletionEvent::Finished on channel"
    );
    eprintln!("Received tokens: {:?}", received_tokens);
    assert!(
        received_tokens.len() >= 4,
        "Expected at least 4 generated tokens before stop token, got {}",
        received_tokens.len()
    );
}

#[tokio::test]
async fn test_end_to_end_subagent_branching_with_independent_sinks() {
    let config = test_micro_model_config();
    let total_blocks = 128;
    let block_size = 16;

    let pool_cfg = KvPoolConfig {
        num_blocks: total_blocks,
        block_size,
        num_layers: config.num_layers,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        dtype: KvDType::Bf16,
    };

    let kv_manager = create_shared_kv_manager(total_blocks, block_size);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();

    let sched_cfg = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 4,
    };

    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());
    let weights = TransformerWeights::reference_test_weights(&config);
    let mut backend = NativeTransformerBackend::with_shared_kv_and_backend(
        weights,
        Arc::new(ReferenceCpuBackend::new()),
        kv_manager.clone(),
    );

    // Parent ticket
    let (sink_parent, mut rx_parent) = ChannelCompletionSink::channel();
    let sink_p_id = spine.register_completion_sink(Arc::new(sink_parent));

    let prompt: PromptHandle = Arc::from([1u32, 15, 30, 45].as_slice());
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 6,
        stop_token_ids: vec![0],
    };

    let parent_id = spine
        .submit_work(prompt, sampling, 2, Some(sink_p_id))
        .unwrap();

    // Step twice to prefill and decode initial token
    let _ = spine.step(&mut backend).await.unwrap();
    let _ = spine.step(&mut backend).await.unwrap();

    // Fork subagent 1 with independent sink
    let (sink_child1, mut rx_child1) = ChannelCompletionSink::channel();
    let sink_c1_id = spine.register_completion_sink(Arc::new(sink_child1));
    let child1_id = parent_id + 100;
    spine
        .fork_subagent(parent_id, child1_id, Some(sink_c1_id))
        .unwrap();

    // Fork subagent 2 with independent sink
    let (sink_child2, mut rx_child2) = ChannelCompletionSink::channel();
    let sink_c2_id = spine.register_completion_sink(Arc::new(sink_child2));
    let child2_id = parent_id + 200;
    spine
        .fork_subagent(parent_id, child2_id, Some(sink_c2_id))
        .unwrap();

    // Verify prefix sharing in physical KV pool right after fork
    {
        let kv = kv_manager.read();
        let m = kv.metrics();
        assert!(
            m.shared_pages >= 1,
            "Parent and subagents must share physical prefix blocks without copy"
        );
    }

    // Run until parent and both subagents complete
    let _ = spine.run_until_complete(&mut backend, 30).await.unwrap();

    // Verify parent received its events
    let mut parent_tokens = Vec::new();
    while let Ok(event) = rx_parent.try_recv() {
        if let CompletionEvent::Token { token, .. } = event {
            parent_tokens.push(token);
        }
    }
    assert!(
        !parent_tokens.is_empty(),
        "Parent must have streamed tokens"
    );

    // Verify child 1 received its events independently
    let mut child1_tokens = Vec::new();
    while let Ok(event) = rx_child1.try_recv() {
        if let CompletionEvent::Token { token, .. } = event {
            child1_tokens.push(token);
        }
    }
    assert!(
        !child1_tokens.is_empty(),
        "Child 1 must have streamed tokens"
    );

    // Verify child 2 received its events independently
    let mut child2_tokens = Vec::new();
    while let Ok(event) = rx_child2.try_recv() {
        if let CompletionEvent::Token { token, .. } = event {
            child2_tokens.push(token);
        }
    }
    assert!(
        !child2_tokens.is_empty(),
        "Child 2 must have streamed tokens"
    );

    eprintln!(
        "Running: {}, Waiting: {}, Preempted: {}, Allocated blocks: {}",
        spine.scheduler.running_count(),
        spine.scheduler.waiting_count(),
        spine.scheduler.preempted_count(),
        kv_manager.read().allocated_block_count()
    );
    {
        let kv = kv_manager.read();
        eprintln!(
            "Active sequences in KV: {}, metrics: {:?}, active blocks: {:?}",
            kv.active_sequence_count(),
            kv.metrics(),
            kv.debug_active_blocks()
        );
    }
    // Verify zero leaked physical pages after all sequences completed
    {
        let kv = kv_manager.read();
        assert_eq!(
            kv.allocated_block_count(),
            0,
            "All physical blocks must be reclaimed upon sequence completion"
        );
    }
}

#[tokio::test]
async fn test_end_to_end_blackwell_hardware_execution_if_available() {
    let gpu_backend = Arc::new(BlackwellGb10Backend::new());
    if !gpu_backend.is_available() {
        eprintln!("Skipping GB10 test: Blackwell hardware not available");
        return;
    }

    let config = test_micro_model_config();
    let total_blocks = 64;
    let block_size = 16;

    let pool_cfg = KvPoolConfig {
        num_blocks: total_blocks,
        block_size,
        num_layers: config.num_layers,
        num_kv_heads: config.num_kv_heads,
        head_dim: config.head_dim,
        dtype: KvDType::Bf16,
    };

    let kv_manager = create_shared_kv_manager(total_blocks, block_size);
    kv_manager.write().attach_tensor_pool(pool_cfg).unwrap();

    let sched_cfg = SchedulerConfig {
        max_batch_size: 16,
        max_batch_tokens: 512,
        max_prefill_tokens: 256,
        prefill_chunk_size: 64,
        chunk_prefill: true,
        watermark_blocks: 4,
    };

    let mut spine = AienRuntimeSpine::new(64, sched_cfg, kv_manager.clone());
    let weights = TransformerWeights::reference_test_weights(&config);
    let mut backend = NativeTransformerBackend::with_shared_kv_and_backend(
        weights,
        gpu_backend.clone(),
        kv_manager,
    );

    let (sink, mut rx) = ChannelCompletionSink::channel();
    let sink_id = spine.register_completion_sink(Arc::new(sink));

    let prompt: PromptHandle = Arc::from([5u32, 12, 33, 77].as_slice());
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 4,
        stop_token_ids: vec![0],
    };

    let seq_id = spine
        .submit_work(prompt, sampling, 1, Some(sink_id))
        .unwrap();

    let _ = spine.run_until_complete(&mut backend, 15).await.unwrap();

    let mut tokens = Vec::new();
    while let Ok(event) = rx.try_recv() {
        if let CompletionEvent::Token { seq_id: sid, token } = event {
            assert_eq!(sid, seq_id);
            tokens.push(token);
        }
    }

    assert_eq!(
        tokens.len(),
        4,
        "Expected 4 tokens generated on Blackwell GPU"
    );
    assert!(
        gpu_backend.kernel_exec_count() > 0,
        "Blackwell GPU kernels must have executed"
    );
    assert_eq!(
        gpu_backend.fallback_count(),
        0,
        "Zero silent CPU fallback allowed on Blackwell hardware"
    );
    eprintln!(
        "Blackwell End-to-End Execution Certified: {} tokens emitted via GPU kernels (count = {})",
        tokens.len(),
        gpu_backend.kernel_exec_count()
    );
}
