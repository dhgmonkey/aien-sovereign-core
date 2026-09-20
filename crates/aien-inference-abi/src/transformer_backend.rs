//! Pure native Rust transformer inference backend executing real forward passes.
//! Dispatches tensor math through the decoupled TensorBackend trait for CPU and Mojo/GB10 execution.

use crate::backend::{ReferenceCpuBackend, TensorBackend};
use crate::blackwell_backend::BlackwellGb10Backend;
use crate::mojo_backend::MojoGb10Backend;
use crate::tensor::{sample_argmax, sample_temperature};
use crate::weights::{LayerKvCache, SequenceState, TransformerWeights};
use crate::{
    AienInferenceBackend, AienUsageReceipt, BranchHandle, ContextHandle, DecodeOutput,
    FinishReason, ModelConfig, ScheduledBatch, StepMetrics,
};
use aien_kv_cache::{create_shared_kv_manager_with_pool, KvDType, KvPoolConfig, SharedKvManager};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);

/// Fully native Rust transformer backend executing real tensor forward computation.
/// Decouples execution orchestration from compute hardware via the TensorBackend trait.
pub struct NativeTransformerBackend {
    pub weights: TransformerWeights,
    pub sequences: HashMap<u64, SequenceState>,
    pub tensor_backend: Arc<dyn TensorBackend>,
    pub kv_manager: Option<SharedKvManager>,
}

impl NativeTransformerBackend {
    /// Creates a new backend with default ReferenceCpuBackend oracle.
    pub fn new(weights: TransformerWeights) -> Self {
        Self {
            weights,
            sequences: HashMap::new(),
            tensor_backend: Arc::new(ReferenceCpuBackend::new()),
            kv_manager: None,
        }
    }

    /// Creates a backend with an explicit TensorBackend implementation.
    pub fn with_backend(
        weights: TransformerWeights,
        tensor_backend: Arc<dyn TensorBackend>,
    ) -> Self {
        Self {
            weights,
            sequences: HashMap::new(),
            tensor_backend,
            kv_manager: None,
        }
    }

    /// Explicit constructor for ReferenceCpuBackend golden oracle.
    pub fn new_reference(weights: TransformerWeights) -> Self {
        Self::new(weights)
    }

    /// Explicit constructor for Mojo GB10 hardware accelerated backend.
    pub fn new_mojo(weights: TransformerWeights) -> Self {
        Self::with_backend(weights, Arc::new(MojoGb10Backend::new()))
    }

    /// Convenience constructor with deterministic reference weights and ReferenceCpuBackend.
    pub fn with_reference_weights(config: &ModelConfig) -> Self {
        let weights = TransformerWeights::reference_test_weights(config);
        Self::new(weights)
    }

    /// Convenience constructor with deterministic reference weights and MojoGb10Backend.
    pub fn with_mojo_backend(config: &ModelConfig) -> Self {
        let weights = TransformerWeights::reference_test_weights(config);
        Self::new_mojo(weights)
    }

    /// Explicit constructor for genuine Blackwell GB10 GPU accelerated backend (sm_121 cuBLAS).
    pub fn new_blackwell(weights: TransformerWeights) -> Self {
        Self::with_backend(weights, Arc::new(BlackwellGb10Backend::new()))
    }

    /// Convenience constructor with deterministic reference weights and BlackwellGb10Backend.
    pub fn with_blackwell_backend(config: &ModelConfig) -> Self {
        let weights = TransformerWeights::reference_test_weights(config);
        Self::new_blackwell(weights)
    }

    /// Constructor attaching a physical reference-counted KV page pool for Copy-on-Write branching.
    pub fn with_paged_kv(
        weights: TransformerWeights,
        total_blocks: usize,
        block_size: usize,
    ) -> Result<Self, String> {
        let pool_cfg = KvPoolConfig::for_model(
            total_blocks,
            block_size,
            weights.config.num_layers,
            weights.config.num_kv_heads,
            weights.config.head_dim,
            KvDType::Fp32,
        );
        let kv_mgr = create_shared_kv_manager_with_pool(total_blocks, block_size, pool_cfg)?;
        Ok(Self {
            weights,
            sequences: HashMap::new(),
            tensor_backend: Arc::new(ReferenceCpuBackend::new()),
            kv_manager: Some(kv_mgr),
        })
    }

    /// Constructor attaching physical paged KV cache with explicit TensorBackend compute hardware.
    pub fn with_paged_kv_backend(
        weights: TransformerWeights,
        tensor_backend: Arc<dyn TensorBackend>,
        total_blocks: usize,
        block_size: usize,
    ) -> Result<Self, String> {
        let pool_cfg = KvPoolConfig::for_model(
            total_blocks,
            block_size,
            weights.config.num_layers,
            weights.config.num_kv_heads,
            weights.config.head_dim,
            KvDType::Fp32,
        );
        let kv_mgr = create_shared_kv_manager_with_pool(total_blocks, block_size, pool_cfg)?;
        Ok(Self {
            weights,
            sequences: HashMap::new(),
            tensor_backend,
            kv_manager: Some(kv_mgr),
        })
    }

    /// Constructor attaching an existing shared KV manager and explicit TensorBackend compute hardware.
    pub fn with_shared_kv_and_backend(
        weights: TransformerWeights,
        tensor_backend: Arc<dyn TensorBackend>,
        kv_manager: SharedKvManager,
    ) -> Self {
        Self {
            weights,
            sequences: HashMap::new(),
            tensor_backend,
            kv_manager: Some(kv_manager),
        }
    }

    /// Creates a new immutable root context from prompt tokens.
    pub fn prefill_sequence(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, String> {
        if prompt_tokens.is_empty() {
            return Err("Cannot prefill empty prompt".to_string());
        }

        if let Some(kv_mgr) = &self.kv_manager {
            kv_mgr.write().allocate_sequence(seq_id, prompt_tokens)?;
        }

        let num_layers = self.weights.config.num_layers;
        let seq = self
            .sequences
            .entry(seq_id)
            .or_insert_with(|| SequenceState {
                tokens: Vec::new(),
                layers: vec![LayerKvCache::default(); num_layers],
            });

        let last_hidden = Self::prefill_prompt_layer_by_layer_paged(
            &self.weights,
            &*self.tensor_backend,
            prompt_tokens,
            seq,
            seq_id,
            self.kv_manager.as_ref(),
        );

        let logits = Self::compute_logits_impl(&self.weights, &*self.tensor_backend, &last_hidden);
        Ok(logits)
    }

    pub fn create_context(&mut self, prompt_tokens: &[u32]) -> Result<ContextHandle, String> {
        if prompt_tokens.is_empty() {
            return Err("Cannot create context with empty prompt".to_string());
        }
        let id = NEXT_HANDLE_ID.fetch_add(1, Ordering::SeqCst);
        let handle = ContextHandle(id);

        if let Some(kv_mgr) = &self.kv_manager {
            kv_mgr.write().allocate_sequence(handle.0, prompt_tokens)?;
        }

        let num_layers = self.weights.config.num_layers;
        let seq = self
            .sequences
            .entry(handle.0)
            .or_insert_with(|| SequenceState {
                tokens: Vec::new(),
                layers: vec![LayerKvCache::default(); num_layers],
            });

        Self::prefill_prompt_layer_by_layer_paged(
            &self.weights,
            &*self.tensor_backend,
            prompt_tokens,
            seq,
            handle.0,
            self.kv_manager.as_ref(),
        );

        Ok(handle)
    }

    /// Forks an existing context into an independently decoding branch with zero KV copying.
    pub fn fork_context(&mut self, parent: ContextHandle) -> Result<BranchHandle, String> {
        let parent_seq = self
            .sequences
            .get(&parent.0)
            .ok_or_else(|| format!("Parent context {} not found", parent.0))?;

        let child_id = NEXT_HANDLE_ID.fetch_add(1, Ordering::SeqCst);
        let child = BranchHandle(child_id);

        let child_seq = if self.kv_manager.is_some() {
            SequenceState {
                tokens: parent_seq.tokens.clone(),
                layers: Vec::new(),
            }
        } else {
            parent_seq.clone()
        };

        if let Some(kv_mgr) = &self.kv_manager {
            kv_mgr.write().fork_context(parent.0, child.0)?;
        }

        self.sequences.insert(child.0, child_seq);
        Ok(child)
    }

    /// Releases a branch and immediately reclaims its private COW pages.
    pub fn release_branch(&mut self, branch: BranchHandle) -> Result<(), String> {
        self.sequences.remove(&branch.0);
        if let Some(kv_mgr) = &self.kv_manager {
            let _ = kv_mgr.write().release_branch(branch.0);
        }
        Ok(())
    }

    /// Executes one autoregressive decode step for a branch, triggering COW if tail page is shared.
    pub fn decode_branch_step(&mut self, branch: BranchHandle) -> Result<(u32, Vec<f32>), String> {
        let seq = self
            .sequences
            .get_mut(&branch.0)
            .ok_or_else(|| format!("Branch {} not found", branch.0))?;

        let pos = seq.tokens.len().saturating_sub(1);
        let last_token = *seq.tokens.last().unwrap_or(&1);

        let hidden = Self::forward_token_impl_paged(
            &self.weights,
            &*self.tensor_backend,
            last_token,
            pos,
            seq,
            branch.0,
            self.kv_manager.as_ref(),
        );

        let logits = Self::compute_logits_impl(&self.weights, &*self.tensor_backend, &hidden);
        let (sampled_tok, _) = sample_argmax(&logits);
        seq.tokens.push(sampled_tok);

        Ok((sampled_tok, logits))
    }

    /// Generates tokens autoregressively, invoking a callback for each generated token.
    pub fn generate_tokens_streaming<F>(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        mut on_token: F,
    ) -> Result<(), String>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt_tokens.is_empty() {
            return Err("Cannot generate from empty prompt".to_string());
        }

        let logits = self.prefill_sequence(seq_id, prompt_tokens)?;

        let (first_tok, _) = if temperature <= 0.001 {
            sample_argmax(&logits)
        } else {
            sample_temperature(&logits, temperature, seq_id)
        };

        if stop_tokens.contains(&first_tok) {
            self.release_sequence(seq_id);
            return Ok(());
        }

        if !on_token(first_tok) {
            self.release_sequence(seq_id);
            return Ok(());
        }

        if let Some(seq) = self.sequences.get_mut(&seq_id) {
            seq.tokens.push(first_tok);
        }

        for step in 1..max_tokens {
            let seq = match self.sequences.get_mut(&seq_id) {
                Some(s) => s,
                None => break,
            };

            let pos = seq.tokens.len().saturating_sub(1);
            let last_token = *seq.tokens.last().unwrap_or(&1);

            let hidden = Self::forward_token_impl_paged(
                &self.weights,
                &*self.tensor_backend,
                last_token,
                pos,
                seq,
                seq_id,
                self.kv_manager.as_ref(),
            );

            let next_logits =
                Self::compute_logits_impl(&self.weights, &*self.tensor_backend, &hidden);
            let (next_tok, _) = if temperature <= 0.001 {
                sample_argmax(&next_logits)
            } else {
                sample_temperature(&next_logits, temperature, seq_id.wrapping_add(step as u64))
            };

            if stop_tokens.contains(&next_tok) {
                break;
            }

            if !on_token(next_tok) {
                break;
            }

            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                seq.tokens.push(next_tok);
            }
        }

        self.release_sequence(seq_id);
        Ok(())
    }

    /// Generates tokens autoregressively from a sequence of prompt tokens.
    pub fn generate_tokens(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
    ) -> Result<Vec<u32>, String> {
        let mut generated = Vec::with_capacity(max_tokens);
        self.generate_tokens_streaming(
            seq_id,
            prompt_tokens,
            max_tokens,
            temperature,
            stop_tokens,
            |tok| {
                generated.push(tok);
                true
            },
        )?;
        Ok(generated)
    }

    /// Releases a sequence from memory and frees associated KV cache resources.
    pub fn release_sequence(&mut self, seq_id: u64) {
        self.sequences.remove(&seq_id);
        if let Some(kv_mgr) = &self.kv_manager {
            let _ = kv_mgr.write().release_branch(seq_id);
        }
    }

    /// Loads TinyLlama weights from safetensors with optional paged KV cache and compute backend.
    pub fn load_tinyllama_safetensors<P: AsRef<std::path::Path>>(
        checkpoint_path: P,
        tensor_backend: Option<Arc<dyn TensorBackend>>,
        total_blocks: Option<usize>,
    ) -> Result<Self, String> {
        let config = ModelConfig::tinyllama_1_1b();
        let weights = TransformerWeights::load_from_safetensors(checkpoint_path.as_ref(), &config)
            .map_err(|e| format!("Failed to load safetensors: {}", e))?;

        let backend = tensor_backend.unwrap_or_else(|| {
            let surface = crate::ExecutionSurface::detect();
            if surface.is_accelerated_gpu() {
                Arc::new(BlackwellGb10Backend::new())
            } else {
                Arc::new(ReferenceCpuBackend::new())
            }
        });

        if let Some(blocks) = total_blocks {
            Self::with_paged_kv_backend(weights, backend, blocks, config.block_size)
        } else {
            Ok(Self::with_backend(weights, backend))
        }
    }

    /// Returns a telemetry receipt for a branch, calculating shared pages and physical bytes saved.
    pub fn get_usage_receipt(&self, branch: BranchHandle) -> Result<AienUsageReceipt, String> {
        let kv_mgr = self
            .kv_manager
            .as_ref()
            .ok_or_else(|| "KV manager not initialized on backend".to_string())?
            .read();

        let table = kv_mgr
            .get_block_table(branch.0)
            .ok_or_else(|| format!("Branch {} not found in KV cache", branch.0))?;
        let metrics = kv_mgr.metrics();
        let total_tokens = table.total_tokens;

        let mut shared_pages = 0;
        let mut private_pages = 0;
        for &b in &table.block_ids {
            if let Some(block) = kv_mgr.get_block(b) {
                if block.is_shared {
                    shared_pages += 1;
                } else {
                    private_pages += 1;
                }
            }
        }

        let bytes_per_block = kv_mgr.tensor_pool().map(|p| p.block_bytes()).unwrap_or(0);
        let physical_kv_bytes = (shared_pages + private_pages) * bytes_per_block;

        Ok(AienUsageReceipt {
            prefix_tokens: shared_pages * kv_mgr.block_size(),
            private_tokens: total_tokens.saturating_sub(shared_pages * kv_mgr.block_size()),
            logical_pages: table.block_ids.len(),
            physical_pages: metrics.physical_pages,
            shared_pages,
            private_pages,
            cow_faults: metrics.cow_faults,
            physical_kv_bytes,
            bytes_saved_vs_full_copy: shared_pages * bytes_per_block,
        })
    }

    /// Internal forward pass executing one token forward pass through the transformer.
    pub fn forward_token_impl(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        token_id: u32,
        pos: usize,
        seq_state: &mut SequenceState,
    ) -> Vec<f32> {
        Self::forward_token_impl_paged(weights, backend, token_id, pos, seq_state, 0, None)
    }

    /// Forward pass executing one token forward pass through the transformer with optional paged COW KV.
    pub fn forward_token_impl_paged(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        token_id: u32,
        pos: usize,
        seq_state: &mut SequenceState,
        seq_id: u64,
        kv_manager: Option<&SharedKvManager>,
    ) -> Vec<f32> {
        let hidden_dim = weights.config.hidden_dim();
        let num_heads = weights.config.num_heads;
        let num_kv_heads = weights.config.num_kv_heads;
        let head_dim = weights.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_dim = weights.config.intermediate_dim();
        let eps = weights.config.rms_norm_eps;
        let theta = weights.config.rope_theta;

        let token_idx = (token_id as usize) % weights.config.vocab_size();
        let mut x =
            weights.embed_tokens[token_idx * hidden_dim..(token_idx + 1) * hidden_dim].to_vec();

        let mut x_norm = vec![0.0f32; hidden_dim];
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut attn_proj = vec![0.0f32; hidden_dim];
        let mut post_norm = vec![0.0f32; hidden_dim];
        let mut gate = vec![0.0f32; intermediate_dim];
        let mut up = vec![0.0f32; intermediate_dim];
        let mut activated = vec![0.0f32; intermediate_dim];
        let mut mlp_out = vec![0.0f32; hidden_dim];

        let block_slot = if let Some(mgr) = kv_manager {
            mgr.write().append_token_with_slot(seq_id).ok()
        } else {
            None
        };

        for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
            backend.rmsnorm(&mut x_norm, &x, &layer_w.input_layernorm, eps);
            backend.matmul_vec(&mut q, &x_norm, &layer_w.q_proj, q_dim, hidden_dim);
            backend.matmul_vec(&mut k, &x_norm, &layer_w.k_proj, kv_dim, hidden_dim);
            backend.matmul_vec(&mut v, &x_norm, &layer_w.v_proj, kv_dim, hidden_dim);

            backend.apply_rope(
                &mut q,
                &mut k,
                pos,
                head_dim,
                num_heads,
                num_kv_heads,
                theta,
            );

            if let (Some((block_id, slot)), Some(mgr)) = (block_slot, kv_manager) {
                let _ = mgr
                    .write()
                    .write_explicit_token_kv(block_id, layer_idx, slot, &k, &v);

                let mgr_read = mgr.read();
                if let (Some(pool), Some(table)) =
                    (mgr_read.tensor_pool(), mgr_read.get_block_table(seq_id))
                {
                    let total_tokens = table.total_tokens;
                    backend.paged_attention(
                        &mut attn_out,
                        &q,
                        pool,
                        &table.block_ids,
                        total_tokens,
                        layer_idx,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    );
                }
            } else {
                let kv_cache = &mut seq_state.layers[layer_idx];
                kv_cache.cached_k.push(k.clone());
                kv_cache.cached_v.push(v.clone());
                kv_cache.flat_k.extend_from_slice(&k);
                kv_cache.flat_v.extend_from_slice(&v);

                let seq_len = kv_cache.cached_k.len();
                backend.gqa_attention(
                    &mut attn_out,
                    &q,
                    &kv_cache.flat_k,
                    &kv_cache.flat_v,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }

            backend.matmul_vec(
                &mut attn_proj,
                &attn_out,
                &layer_w.o_proj,
                hidden_dim,
                q_dim,
            );
            for i in 0..hidden_dim {
                x[i] += attn_proj[i];
            }

            backend.rmsnorm(&mut post_norm, &x, &layer_w.post_attention_layernorm, eps);
            backend.matmul_vec(
                &mut gate,
                &post_norm,
                &layer_w.gate_proj,
                intermediate_dim,
                hidden_dim,
            );
            backend.matmul_vec(
                &mut up,
                &post_norm,
                &layer_w.up_proj,
                intermediate_dim,
                hidden_dim,
            );
            backend.swiglu(&mut activated, &gate, &up);
            backend.matmul_vec(
                &mut mlp_out,
                &activated,
                &layer_w.down_proj,
                hidden_dim,
                intermediate_dim,
            );

            for i in 0..hidden_dim {
                x[i] += mlp_out[i];
            }
        }

        let mut x_final = vec![0.0f32; hidden_dim];
        backend.rmsnorm(&mut x_final, &x, &weights.final_norm, eps);
        x_final
    }

    pub fn compute_logits_impl(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        hidden_state: &[f32],
    ) -> Vec<f32> {
        let vocab_size = weights.config.vocab_size();
        let hidden_dim = weights.config.hidden_dim();
        let mut logits = vec![0.0f32; vocab_size];
        backend.compute_logits(
            &mut logits,
            hidden_state,
            &weights.lm_head,
            vocab_size,
            hidden_dim,
        );
        logits
    }

    /// Prefill all prompt tokens layer-by-layer rather than token-by-token.
    pub fn prefill_prompt_layer_by_layer(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        prompt_tokens: &[u32],
        seq_state: &mut SequenceState,
    ) -> Vec<f32> {
        Self::prefill_prompt_layer_by_layer_paged(
            weights,
            backend,
            prompt_tokens,
            seq_state,
            0,
            None,
        )
    }

    /// Prefill all prompt tokens layer-by-layer with optional paged COW KV registration.
    pub fn prefill_prompt_layer_by_layer_paged(
        weights: &TransformerWeights,
        backend: &dyn TensorBackend,
        prompt_tokens: &[u32],
        seq_state: &mut SequenceState,
        seq_id: u64,
        kv_manager: Option<&SharedKvManager>,
    ) -> Vec<f32> {
        let n = prompt_tokens.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            seq_state.tokens.push(prompt_tokens[0]);
            return Self::forward_token_impl_paged(
                weights,
                backend,
                prompt_tokens[0],
                0,
                seq_state,
                seq_id,
                kv_manager,
            );
        }

        let hidden_dim = weights.config.hidden_dim();
        let num_heads = weights.config.num_heads;
        let num_kv_heads = weights.config.num_kv_heads;
        let head_dim = weights.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_dim = weights.config.intermediate_dim();
        let eps = weights.config.rms_norm_eps;
        let theta = weights.config.rope_theta;

        seq_state.tokens.extend_from_slice(prompt_tokens);

        let mut states: Vec<Vec<f32>> = prompt_tokens
            .iter()
            .map(|&tok| {
                let token_idx = (tok as usize) % weights.config.vocab_size();
                let slice =
                    &weights.embed_tokens[token_idx * hidden_dim..(token_idx + 1) * hidden_dim];
                slice.to_vec()
            })
            .collect();

        let mut x_norm_batch = vec![0.0f32; n * hidden_dim];
        let mut q_batch = vec![0.0f32; n * q_dim];
        let mut k_batch = vec![0.0f32; n * kv_dim];
        let mut v_batch = vec![0.0f32; n * kv_dim];
        let mut attn_out_batch = vec![0.0f32; n * q_dim];
        let mut attn_proj_batch = vec![0.0f32; n * hidden_dim];
        let mut post_norm_batch = vec![0.0f32; n * hidden_dim];
        let mut gate_batch = vec![0.0f32; n * intermediate_dim];
        let mut up_batch = vec![0.0f32; n * intermediate_dim];
        let mut act_batch = vec![0.0f32; n * intermediate_dim];
        let mut mlp_out_batch = vec![0.0f32; n * hidden_dim];

        let block_table = kv_manager.and_then(|mgr| mgr.read().get_block_table(seq_id).cloned());

        for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
            let kv_cache = &mut seq_state.layers[layer_idx];

            for t in 0..n {
                let out_slice = &mut x_norm_batch[t * hidden_dim..(t + 1) * hidden_dim];
                backend.rmsnorm(out_slice, &states[t], &layer_w.input_layernorm, eps);
            }

            backend.matmul_batch(
                &mut q_batch,
                &x_norm_batch,
                &layer_w.q_proj,
                n,
                hidden_dim,
                q_dim,
            );
            backend.matmul_batch(
                &mut k_batch,
                &x_norm_batch,
                &layer_w.k_proj,
                n,
                hidden_dim,
                kv_dim,
            );
            backend.matmul_batch(
                &mut v_batch,
                &x_norm_batch,
                &layer_w.v_proj,
                n,
                hidden_dim,
                kv_dim,
            );

            for t in 0..n {
                let q_t = &mut q_batch[t * q_dim..(t + 1) * q_dim];
                let k_t = &mut k_batch[t * kv_dim..(t + 1) * kv_dim];
                let v_t = &v_batch[t * kv_dim..(t + 1) * kv_dim];

                backend.apply_rope(q_t, k_t, t, head_dim, num_heads, num_kv_heads, theta);

                if let (Some(tbl), Some(mgr)) = (&block_table, kv_manager) {
                    let block_size = weights.config.block_size;
                    let block_idx = t / block_size;
                    let block_id = tbl.block_ids[block_idx];
                    let slot = t % block_size;
                    let _ = mgr
                        .write()
                        .write_explicit_token_kv(block_id, layer_idx, slot, k_t, v_t);
                }

                kv_cache.cached_k.push(k_t.to_vec());
                kv_cache.cached_v.push(v_t.to_vec());
                kv_cache.flat_k.extend_from_slice(k_t);
                kv_cache.flat_v.extend_from_slice(v_t);
            }

            for t in 0..n {
                let q_t = &q_batch[t * q_dim..(t + 1) * q_dim];
                let attn_t = &mut attn_out_batch[t * q_dim..(t + 1) * q_dim];
                let seq_len = t + 1;

                backend.gqa_attention(
                    attn_t,
                    q_t,
                    &kv_cache.flat_k,
                    &kv_cache.flat_v,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }

            backend.matmul_batch(
                &mut attn_proj_batch,
                &attn_out_batch,
                &layer_w.o_proj,
                n,
                q_dim,
                hidden_dim,
            );
            for t in 0..n {
                let proj_t = &attn_proj_batch[t * hidden_dim..(t + 1) * hidden_dim];
                for i in 0..hidden_dim {
                    states[t][i] += proj_t[i];
                }
            }

            for t in 0..n {
                let out_slice = &mut post_norm_batch[t * hidden_dim..(t + 1) * hidden_dim];
                backend.rmsnorm(
                    out_slice,
                    &states[t],
                    &layer_w.post_attention_layernorm,
                    eps,
                );
            }

            backend.matmul_batch(
                &mut gate_batch,
                &post_norm_batch,
                &layer_w.gate_proj,
                n,
                hidden_dim,
                intermediate_dim,
            );
            backend.matmul_batch(
                &mut up_batch,
                &post_norm_batch,
                &layer_w.up_proj,
                n,
                hidden_dim,
                intermediate_dim,
            );

            for t in 0..n {
                let gate_t = &gate_batch[t * intermediate_dim..(t + 1) * intermediate_dim];
                let up_t = &up_batch[t * intermediate_dim..(t + 1) * intermediate_dim];
                let act_t = &mut act_batch[t * intermediate_dim..(t + 1) * intermediate_dim];
                backend.swiglu(act_t, gate_t, up_t);
            }

            backend.matmul_batch(
                &mut mlp_out_batch,
                &act_batch,
                &layer_w.down_proj,
                n,
                intermediate_dim,
                hidden_dim,
            );
            for t in 0..n {
                let mlp_t = &mlp_out_batch[t * hidden_dim..(t + 1) * hidden_dim];
                for i in 0..hidden_dim {
                    states[t][i] += mlp_t[i];
                }
            }
        }

        if kv_manager.is_some() {
            seq_state.layers.clear();
        }

        let last_x = &states[n - 1];
        let mut x_final = vec![0.0f32; hidden_dim];
        backend.rmsnorm(&mut x_final, last_x, &weights.final_norm, eps);
        x_final
    }

    /// Single-token forward pass executing through the configured TensorBackend trait.
    pub fn forward_token(
        &self,
        token_id: u32,
        pos: usize,
        seq_state: &mut SequenceState,
    ) -> Vec<f32> {
        Self::forward_token_impl(
            &self.weights,
            &*self.tensor_backend,
            token_id,
            pos,
            seq_state,
        )
    }

    /// Computes logits projection executing through the configured TensorBackend trait.
    pub fn compute_logits(&self, hidden_state: &[f32]) -> Vec<f32> {
        Self::compute_logits_impl(&self.weights, &*self.tensor_backend, hidden_state)
    }

    /// Forward pass executing one token decode step across a batch of D sequences concurrently.
    /// Employs batched GEMM (M=D) for all linear projections and batched paged attention.
    pub fn forward_decode_batch(&mut self, decode_req_ids: &[u64]) -> Vec<DecodeOutput> {
        if decode_req_ids.is_empty() {
            return Vec::new();
        }

        let num_layers = self.weights.config.num_layers;
        let hidden_dim = self.weights.config.hidden_dim();
        let num_heads = self.weights.config.num_heads;
        let num_kv_heads = self.weights.config.num_kv_heads;
        let head_dim = self.weights.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let intermediate_dim = self.weights.config.intermediate_dim();
        let eps = self.weights.config.rms_norm_eps;
        let theta = self.weights.config.rope_theta;
        let vocab_size = self.weights.config.vocab_size();

        // 1. Gather active sequence requests and their current positions / last tokens
        let mut valid_reqs = Vec::with_capacity(decode_req_ids.len());
        for &req_id in decode_req_ids {
            let seq = self
                .sequences
                .entry(req_id)
                .or_insert_with(|| SequenceState {
                    tokens: Vec::new(),
                    layers: vec![LayerKvCache::default(); num_layers],
                });
            let pos = seq.tokens.len().saturating_sub(1);
            let last_token = *seq.tokens.last().unwrap_or(&1);
            valid_reqs.push((req_id, last_token, pos));
        }

        let d = valid_reqs.len();
        if d == 0 {
            return Vec::new();
        }

        // 2. Allocate batch activations
        let mut x = vec![0.0f32; d * hidden_dim];
        let mut x_norm = vec![0.0f32; d * hidden_dim];
        let mut q_batch = vec![0.0f32; d * q_dim];
        let mut k_batch = vec![0.0f32; d * kv_dim];
        let mut v_batch = vec![0.0f32; d * kv_dim];
        let mut attn_out_batch = vec![0.0f32; d * q_dim];
        let mut attn_proj_batch = vec![0.0f32; d * hidden_dim];
        let mut post_norm_batch = vec![0.0f32; d * hidden_dim];
        let mut gate_batch = vec![0.0f32; d * intermediate_dim];
        let mut up_batch = vec![0.0f32; d * intermediate_dim];
        let mut act_batch = vec![0.0f32; d * intermediate_dim];
        let mut mlp_out_batch = vec![0.0f32; d * hidden_dim];
        let mut logits_batch = vec![0.0f32; d * vocab_size];

        // 3. Populate initial embeddings
        for (i, &(_seq_id, last_token, _pos)) in valid_reqs.iter().enumerate() {
            self.weights
                .embed_token(last_token, &mut x[i * hidden_dim..(i + 1) * hidden_dim]);
        }

        // 4. Reserve append slots in KV cache for this decode step
        let block_slots: Vec<Option<(usize, usize)>> = if let Some(mgr) = &self.kv_manager {
            let mut write_mgr = mgr.write();
            valid_reqs
                .iter()
                .map(|&(seq_id, _, _)| write_mgr.append_token_with_slot(seq_id).ok())
                .collect()
        } else {
            vec![None; d]
        };

        // 5. Transformer layer loop
        for (layer_idx, layer_w) in self.weights.layers.iter().enumerate() {
            // a. Pre-attention RMSNorm
            for i in 0..d {
                self.tensor_backend.rmsnorm(
                    &mut x_norm[i * hidden_dim..(i + 1) * hidden_dim],
                    &x[i * hidden_dim..(i + 1) * hidden_dim],
                    &layer_w.input_layernorm,
                    eps,
                );
            }

            // b. Batched Q, K, V projections (M = D)
            self.tensor_backend.matmul_batch(
                &mut q_batch,
                &x_norm,
                &layer_w.q_proj,
                d,
                hidden_dim,
                q_dim,
            );
            self.tensor_backend.matmul_batch(
                &mut k_batch,
                &x_norm,
                &layer_w.k_proj,
                d,
                hidden_dim,
                kv_dim,
            );
            self.tensor_backend.matmul_batch(
                &mut v_batch,
                &x_norm,
                &layer_w.v_proj,
                d,
                hidden_dim,
                kv_dim,
            );

            // c. RoPE across all D sequences at their respective positions
            for (i, &(_seq_id, _tok, pos)) in valid_reqs.iter().enumerate() {
                self.tensor_backend.apply_rope(
                    &mut q_batch[i * q_dim..(i + 1) * q_dim],
                    &mut k_batch[i * kv_dim..(i + 1) * kv_dim],
                    pos,
                    head_dim,
                    num_heads,
                    num_kv_heads,
                    theta,
                );
            }

            // d. Write K & V into KV cache
            if let Some(mgr) = &self.kv_manager {
                let mut write_mgr = mgr.write();
                for (i, slot_opt) in block_slots.iter().enumerate() {
                    if let Some((block_id, slot)) = *slot_opt {
                        let k_slice = &k_batch[i * kv_dim..(i + 1) * kv_dim];
                        let v_slice = &v_batch[i * kv_dim..(i + 1) * kv_dim];
                        let _ = write_mgr
                            .write_explicit_token_kv(block_id, layer_idx, slot, k_slice, v_slice);
                    }
                }
            } else {
                for (i, &(seq_id, _, _)) in valid_reqs.iter().enumerate() {
                    if let Some(seq) = self.sequences.get_mut(&seq_id) {
                        let kv_cache = &mut seq.layers[layer_idx];
                        let k_slice = &k_batch[i * kv_dim..(i + 1) * kv_dim];
                        let v_slice = &v_batch[i * kv_dim..(i + 1) * kv_dim];
                        kv_cache.cached_k.push(k_slice.to_vec());
                        kv_cache.cached_v.push(v_slice.to_vec());
                        kv_cache.flat_k.extend_from_slice(k_slice);
                        kv_cache.flat_v.extend_from_slice(v_slice);
                    }
                }
            }

            // e. Attention execution
            let paged_computed = if let Some(mgr) = &self.kv_manager {
                let read_mgr = mgr.read();
                if let Some(pool) = read_mgr.tensor_pool() {
                    let mut block_tables = Vec::with_capacity(d);
                    let mut context_lens = Vec::with_capacity(d);
                    let mut max_blocks = 1;
                    for &(seq_id, _, _pos) in &valid_reqs {
                        if let Some(table) = read_mgr.get_block_table(seq_id) {
                            if table.block_ids.len() > max_blocks {
                                max_blocks = table.block_ids.len();
                            }
                            block_tables.push(table.block_ids.clone());
                            context_lens.push(table.total_tokens as i32);
                        } else {
                            block_tables.push(Vec::new());
                            context_lens.push(0);
                        }
                    }

                    let mut flat_block_tables = vec![-1i32; d * max_blocks];
                    for (i, bt) in block_tables.iter().enumerate() {
                        for (j, &blk) in bt.iter().enumerate() {
                            flat_block_tables[i * max_blocks + j] = blk as i32;
                        }
                    }

                    self.tensor_backend.paged_attention_batch(
                        &mut attn_out_batch,
                        &q_batch,
                        pool,
                        &flat_block_tables,
                        &context_lens,
                        max_blocks,
                        d,
                        layer_idx,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    );
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if !paged_computed {
                for (i, &(seq_id, _, _pos)) in valid_reqs.iter().enumerate() {
                    let q_slice = &q_batch[i * q_dim..(i + 1) * q_dim];
                    let out_slice = &mut attn_out_batch[i * q_dim..(i + 1) * q_dim];
                    if let Some(seq) = self.sequences.get(&seq_id) {
                        let kv_cache = &seq.layers[layer_idx];
                        let seq_len = kv_cache.cached_k.len();
                        self.tensor_backend.gqa_attention(
                            out_slice,
                            q_slice,
                            &kv_cache.flat_k,
                            &kv_cache.flat_v,
                            seq_len,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                        );
                    }
                }
            }

            // f. Attention output projection (M = D)
            self.tensor_backend.matmul_batch(
                &mut attn_proj_batch,
                &attn_out_batch,
                &layer_w.o_proj,
                d,
                q_dim,
                hidden_dim,
            );

            // g. Residual connection
            for idx in 0..x.len() {
                x[idx] += attn_proj_batch[idx];
            }

            // h. Post-attention RMSNorm
            for i in 0..d {
                self.tensor_backend.rmsnorm(
                    &mut post_norm_batch[i * hidden_dim..(i + 1) * hidden_dim],
                    &x[i * hidden_dim..(i + 1) * hidden_dim],
                    &layer_w.post_attention_layernorm,
                    eps,
                );
            }

            // i. MLP Gate & Up projections (M = D)
            self.tensor_backend.matmul_batch(
                &mut gate_batch,
                &post_norm_batch,
                &layer_w.gate_proj,
                d,
                hidden_dim,
                intermediate_dim,
            );
            self.tensor_backend.matmul_batch(
                &mut up_batch,
                &post_norm_batch,
                &layer_w.up_proj,
                d,
                hidden_dim,
                intermediate_dim,
            );

            // j. SwiGLU activation
            for i in 0..d {
                self.tensor_backend.swiglu(
                    &mut act_batch[i * intermediate_dim..(i + 1) * intermediate_dim],
                    &gate_batch[i * intermediate_dim..(i + 1) * intermediate_dim],
                    &up_batch[i * intermediate_dim..(i + 1) * intermediate_dim],
                );
            }

            // k. MLP Down projection (M = D)
            self.tensor_backend.matmul_batch(
                &mut mlp_out_batch,
                &act_batch,
                &layer_w.down_proj,
                d,
                intermediate_dim,
                hidden_dim,
            );

            // l. Residual connection
            for idx in 0..x.len() {
                x[idx] += mlp_out_batch[idx];
            }
        }

        // 6. Final RMSNorm
        for i in 0..d {
            self.tensor_backend.rmsnorm(
                &mut x_norm[i * hidden_dim..(i + 1) * hidden_dim],
                &x[i * hidden_dim..(i + 1) * hidden_dim],
                &self.weights.final_norm,
                eps,
            );
        }

        // 7. Batched Vocabulary Logits Projection (M = D)
        self.tensor_backend.matmul_batch(
            &mut logits_batch,
            &x_norm,
            &self.weights.lm_head,
            d,
            hidden_dim,
            vocab_size,
        );

        // 8. Sampling and output emission
        let mut outputs = Vec::with_capacity(d);
        let stop_tokens = [
            crate::tokenizer::TinyLlamaTokenizer::UNK_TOKEN_ID,
            crate::tokenizer::TinyLlamaTokenizer::BOS_TOKEN_ID,
            crate::tokenizer::TinyLlamaTokenizer::EOS_TOKEN_ID,
        ];

        for (i, &(seq_id, _, _)) in valid_reqs.iter().enumerate() {
            let logits = &logits_batch[i * vocab_size..(i + 1) * vocab_size];
            let (sampled_tok, logprob) = sample_argmax(logits);

            if let Some(seq) = self.sequences.get_mut(&seq_id) {
                seq.tokens.push(sampled_tok);

                if stop_tokens.contains(&sampled_tok) {
                    outputs.push(DecodeOutput::Finished {
                        request_id: seq_id,
                        reason: FinishReason::StopToken,
                        total_tokens: seq.tokens.len(),
                    });
                } else {
                    outputs.push(DecodeOutput::Token {
                        request_id: seq_id,
                        token_id: sampled_tok,
                        logprob: Some(logprob),
                    });
                }
            }
        }

        outputs
    }
}

#[async_trait]
impl AienInferenceBackend for NativeTransformerBackend {
    async fn load_model(&mut self, config: &ModelConfig) -> Result<(), String> {
        self.weights = TransformerWeights::reference_test_weights(config);
        self.sequences.clear();
        Ok(())
    }

    fn manages_kv_cache(&self) -> bool {
        self.kv_manager.is_some()
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let t0 = std::time::Instant::now();
        let mut outputs = Vec::new();
        let mut prefill_tokens = 0;
        let num_layers = self.weights.config.num_layers;

        // 1. Prefill Requests
        for req in &batch.prefill_requests {
            prefill_tokens += req.prompt_tokens.len();

            if let Some(kv_mgr) = &self.kv_manager {
                let mut mgr = kv_mgr.write();
                if mgr.get_block_table(req.request_id).is_none() {
                    let _ = mgr.allocate_sequence(req.request_id, &req.prompt_tokens);
                }
            }

            let seq = self
                .sequences
                .entry(req.request_id)
                .or_insert_with(|| SequenceState {
                    tokens: Vec::new(),
                    layers: vec![LayerKvCache::default(); num_layers],
                });

            let last_hidden = Self::prefill_prompt_layer_by_layer_paged(
                &self.weights,
                &*self.tensor_backend,
                &req.prompt_tokens,
                seq,
                req.request_id,
                self.kv_manager.as_ref(),
            );

            let logits =
                Self::compute_logits_impl(&self.weights, &*self.tensor_backend, &last_hidden);
            let (sampled_tok, logprob) = if req.sampling_params.temperature <= 0.001 {
                sample_argmax(&logits)
            } else {
                sample_temperature(&logits, req.sampling_params.temperature, req.request_id)
            };

            seq.tokens.push(sampled_tok);

            outputs.push(DecodeOutput::Token {
                request_id: req.request_id,
                token_id: sampled_tok,
                logprob: Some(logprob),
            });
        }

        // 2. Decode Requests (Batched across D active sequences)
        if !batch.decode_requests.is_empty() {
            let decode_outputs = self.forward_decode_batch(&batch.decode_requests);
            outputs.extend(decode_outputs);
        }

        let elapsed_us = t0.elapsed().as_micros() as u64;
        let metrics = StepMetrics {
            prefill_tokens_processed: prefill_tokens,
            decode_tokens_emitted: batch.decode_requests.len() + batch.prefill_requests.len(),
            step_latency_us: elapsed_us,
            active_kv_blocks: batch.block_tables.values().map(|v| v.len()).sum(),
        };

        Ok((outputs, metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_constructors() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            ..Default::default()
        };

        let cpu_backend = NativeTransformerBackend::with_reference_weights(&config);
        assert_eq!(cpu_backend.tensor_backend.name(), "ReferenceCpuBackend");

        let mojo_backend = NativeTransformerBackend::with_mojo_backend(&config);
        assert!(mojo_backend
            .tensor_backend
            .name()
            .starts_with("NativeCpuBackend"));

        let blackwell_backend = NativeTransformerBackend::with_blackwell_backend(&config);
        assert!(blackwell_backend
            .tensor_backend
            .name()
            .starts_with("BlackwellGb10Backend"));
    }

    #[test]
    fn test_forward_token_and_logits_execution() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            ..Default::default()
        };

        let backend = NativeTransformerBackend::with_reference_weights(&config);
        let mut seq_state = SequenceState {
            tokens: vec![1],
            layers: vec![LayerKvCache::default(); config.num_layers],
        };

        let hidden = backend.forward_token(1, 0, &mut seq_state);
        assert_eq!(hidden.len(), config.hidden_dim());

        let logits = backend.compute_logits(&hidden);
        assert_eq!(logits.len(), config.vocab_size());
    }

    #[test]
    fn test_paged_cow_context_and_branch_fork() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            block_size: 16,
            ..Default::default()
        };

        let weights = TransformerWeights::reference_test_weights(&config);
        let mut backend = NativeTransformerBackend::with_paged_kv(weights, 200, 16).unwrap();

        // 1. Create parent context with 30 prompt tokens (2 blocks)
        let prompt: Vec<u32> = (0..30).collect();
        let parent = backend.create_context(&prompt).unwrap();

        // 2. Fork 4 branches
        let b1 = backend.fork_context(parent).unwrap();
        let b2 = backend.fork_context(parent).unwrap();
        let b3 = backend.fork_context(parent).unwrap();
        let b4 = backend.fork_context(parent).unwrap();

        // Initial receipt before decode: 2 shared pages
        let r_init = backend.get_usage_receipt(b1).unwrap();
        assert_eq!(r_init.shared_pages, 2);
        assert_eq!(r_init.private_pages, 0);

        // 3. Decode step for each branch -> triggers COW divergence on partial block 1
        let (t1, _) = backend.decode_branch_step(b1).unwrap();
        let (t2, _) = backend.decode_branch_step(b2).unwrap();
        let (t3, _) = backend.decode_branch_step(b3).unwrap();
        let (t4, _) = backend.decode_branch_step(b4).unwrap();

        // Tokens generated
        assert_eq!(t1, t2); // Same deterministic greedy choice initially
        assert_eq!(t3, t4);

        let r_after = backend.get_usage_receipt(b1).unwrap();
        assert_eq!(r_after.cow_faults, 4); // All 4 branches COWed tail page
        assert_eq!(r_after.shared_pages, 1); // Prefix block 0 remains shared
        assert_eq!(r_after.private_pages, 1); // Divergent block 1 is private

        // 4. Release branches
        backend.release_branch(b1).unwrap();
        backend.release_branch(b2).unwrap();
        backend.release_branch(b3).unwrap();
        backend.release_branch(b4).unwrap();
    }

    #[test]
    fn test_native_transformer_generate_tokens() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            block_size: 16,
            ..Default::default()
        };
        let mut backend = NativeTransformerBackend::with_reference_weights(&config);
        let prompt = vec![1, 5, 9];
        let stop_tokens = vec![0];
        let generated = backend
            .generate_tokens(555, &prompt, 4, 0.0, &stop_tokens)
            .unwrap();
        assert_eq!(generated.len(), 4);
        assert!(!generated.contains(&0));
        assert_eq!(backend.sequences.len(), 0);
    }

    #[test]
    fn test_native_transformer_generate_tokens_streaming() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            hidden_dim: 64,
            intermediate_dim: 128,
            vocab_size: 256,
            block_size: 16,
            ..Default::default()
        };
        let mut backend = NativeTransformerBackend::with_reference_weights(&config);
        let prompt = vec![1, 5, 9];
        let stop_tokens = vec![0];
        let mut streamed = Vec::new();
        backend
            .generate_tokens_streaming(777, &prompt, 4, 0.0, &stop_tokens, |tok| {
                streamed.push(tok);
                true
            })
            .unwrap();
        assert_eq!(streamed.len(), 4);
        assert!(!streamed.contains(&0));
        assert_eq!(backend.sequences.len(), 0);
    }
}
