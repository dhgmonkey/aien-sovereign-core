//! Native Paged KV-Cache Manager with physical unified-memory tensors,
//! reference-counted copy-on-write branching, and prefix caching.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub type BlockId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvDType {
    Fp4,
    Fp8,
    Bf16,
    Fp16,
    Fp32,
}

impl KvDType {
    pub fn bytes_per_element(&self) -> f32 {
        match self {
            KvDType::Fp4 => 0.5,
            KvDType::Fp8 => 1.0,
            KvDType::Bf16 | KvDType::Fp16 => 2.0,
            KvDType::Fp32 => 4.0,
        }
    }
}

/// Exact IEEE-754 / bfloat16 round-to-nearest-even conversion.
#[inline]
pub fn f32_to_bf16_bits(val: f32) -> u16 {
    let u = val.to_bits();
    let lsb = (u >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((u.wrapping_add(rounding_bias)) >> 16) as u16
}

#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvPoolConfig {
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: KvDType,
}

impl KvPoolConfig {
    pub fn for_tinyllama(num_blocks: usize, block_size: usize, dtype: KvDType) -> Self {
        Self {
            num_blocks,
            block_size,
            num_layers: 22,
            num_kv_heads: 4,
            head_dim: 64,
            dtype,
        }
    }

    pub fn for_qwen2_5_7b(num_blocks: usize, block_size: usize, dtype: KvDType) -> Self {
        Self {
            num_blocks,
            block_size,
            num_layers: 28,
            num_kv_heads: 4,
            head_dim: 128,
            dtype,
        }
    }

    pub fn for_nemotron_30b(num_blocks: usize, block_size: usize, dtype: KvDType) -> Self {
        Self {
            num_blocks,
            block_size,
            num_layers: 48,
            num_kv_heads: 8,
            head_dim: 128,
            dtype,
        }
    }

    pub fn for_model(
        num_blocks: usize,
        block_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: KvDType,
    ) -> Self {
        Self {
            num_blocks,
            block_size,
            num_layers,
            num_kv_heads,
            head_dim,
            dtype,
        }
    }

    pub fn bytes_per_block(&self) -> usize {
        let elements = 2 * self.num_layers * self.block_size * self.num_kv_heads * self.head_dim;
        (elements as f32 * self.dtype.bytes_per_element()).ceil() as usize
    }

    pub fn total_bytes(&self) -> usize {
        self.num_blocks * self.bytes_per_block()
    }

    pub fn layout(&self) -> Result<KvLayout, String> {
        KvLayout::from_config(self)
    }

    pub fn layout_desc(&self) -> Result<KvLayoutDesc, String> {
        Ok(self.layout()?.to_desc())
    }
}

/// Explicit C-ABI layout descriptor matching CUDA kernel parameterization.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvLayoutDesc {
    pub block_stride_bytes: u64,
    pub layer_stride_bytes: u64,
    pub kv_plane_stride_bytes: u64,
    pub token_stride_bytes: u64,
    pub head_stride_bytes: u64,

    pub pool_bytes: u64,
    pub num_blocks: u32,
    pub num_layers: u32,
    pub block_size: u32,
}

/// Canonical physical layout definition owning authoritative stride mathematics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvLayout {
    pub block_stride_bytes: usize,
    pub layer_stride_bytes: usize,
    pub kv_plane_stride_bytes: usize,
    pub token_stride_bytes: usize,
    pub head_stride_bytes: usize,
    pub element_bytes: usize,
    pub total_bytes: usize,
    pub num_blocks: usize,
    pub num_layers: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KvLayout {
    pub fn for_bf16(config: &KvPoolConfig) -> Result<Self, String> {
        if config.dtype != KvDType::Bf16 {
            return Err("Expected BF16 dtype".to_string());
        }
        Self::from_config(config)
    }

    pub fn from_config(config: &KvPoolConfig) -> Result<Self, String> {
        let element_bytes = match config.dtype {
            KvDType::Bf16 | KvDType::Fp16 => 2usize,
            KvDType::Fp32 => 4usize,
            KvDType::Fp8 => 1usize,
            KvDType::Fp4 => {
                return Err("Fp4 sub-byte stride requires packed layout".to_string());
            }
        };

        let head_stride_bytes = config.head_dim * element_bytes;
        let token_stride_bytes = config.num_kv_heads * head_stride_bytes;
        let kv_plane_stride_bytes = config.block_size * token_stride_bytes;
        let layer_stride_bytes = 2 * kv_plane_stride_bytes;
        let block_stride_bytes = config.num_layers * layer_stride_bytes;
        let total_bytes = config.num_blocks * block_stride_bytes;

        Ok(Self {
            block_stride_bytes,
            layer_stride_bytes,
            kv_plane_stride_bytes,
            token_stride_bytes,
            head_stride_bytes,
            element_bytes,
            total_bytes,
            num_blocks: config.num_blocks,
            num_layers: config.num_layers,
            block_size: config.block_size,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
        })
    }

    #[inline]
    pub fn element_offset(
        &self,
        block: usize,
        layer: usize,
        is_value: bool,
        token: usize,
        head: usize,
        dim: usize,
    ) -> usize {
        assert!(block < self.num_blocks, "Block {} out of range", block);
        assert!(layer < self.num_layers, "Layer {} out of range", layer);
        assert!(token < self.block_size, "Token {} out of range", token);
        assert!(head < self.num_kv_heads, "Head {} out of range", head);
        assert!(dim < self.head_dim, "Dim {} out of range", dim);

        block * self.block_stride_bytes
            + layer * self.layer_stride_bytes
            + usize::from(is_value) * self.kv_plane_stride_bytes
            + token * self.token_stride_bytes
            + head * self.head_stride_bytes
            + dim * self.element_bytes
    }

    #[inline]
    pub fn token_plane_offset(
        &self,
        block: usize,
        layer: usize,
        is_value: bool,
        token: usize,
    ) -> usize {
        assert!(block < self.num_blocks, "Block {} out of range", block);
        assert!(layer < self.num_layers, "Layer {} out of range", layer);
        assert!(token < self.block_size, "Token {} out of range", token);

        block * self.block_stride_bytes
            + layer * self.layer_stride_bytes
            + usize::from(is_value) * self.kv_plane_stride_bytes
            + token * self.token_stride_bytes
    }

    #[inline]
    pub fn block_offset(&self, block: usize) -> usize {
        assert!(block < self.num_blocks, "Block {} out of range", block);
        block * self.block_stride_bytes
    }

    pub fn to_desc(&self) -> KvLayoutDesc {
        KvLayoutDesc {
            block_stride_bytes: self.block_stride_bytes as u64,
            layer_stride_bytes: self.layer_stride_bytes as u64,
            kv_plane_stride_bytes: self.kv_plane_stride_bytes as u64,
            token_stride_bytes: self.token_stride_bytes as u64,
            head_stride_bytes: self.head_stride_bytes as u64,
            pool_bytes: self.total_bytes as u64,
            num_blocks: self.num_blocks as u32,
            num_layers: self.num_layers as u32,
            block_size: self.block_size as u32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferRegion {
    pub address: aien_platform::DeviceAddress,
    pub offset: usize,
    pub len: usize,
}

/// Fallback standard host buffer implementing UnifiedBuffer for testing.
pub struct HeapUnifiedBuffer {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    len: usize,
}

unsafe impl Send for HeapUnifiedBuffer {}
unsafe impl Sync for HeapUnifiedBuffer {}

impl HeapUnifiedBuffer {
    pub fn allocate(size: usize, align: usize) -> Result<Self, String> {
        let align = align.max(std::mem::align_of::<usize>());
        let layout = std::alloc::Layout::from_size_align(size, align)
            .map_err(|e| format!("Invalid layout: {}", e))?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(format!("Failed to allocate {} bytes on heap", size));
        }
        Ok(Self {
            ptr,
            layout,
            len: size,
        })
    }
}

impl aien_platform::UnifiedBuffer for HeapUnifiedBuffer {
    fn len(&self) -> usize {
        self.len
    }
    fn device_address(&self) -> aien_platform::DeviceAddress {
        aien_platform::DeviceAddress(self.ptr as u64)
    }
    fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for HeapUnifiedBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() > 0 {
            unsafe {
                std::alloc::dealloc(self.ptr, self.layout);
            }
        }
    }
}

/// Physical unified-memory KV tensor pool owning layout interpretation over an underlying UnifiedBuffer.
pub struct UnifiedKvTensorPool<B: aien_platform::UnifiedBuffer = HeapUnifiedBuffer> {
    config: KvPoolConfig,
    layout: KvLayout,
    buffer: B,
}

unsafe impl<B: aien_platform::UnifiedBuffer + Send> Send for UnifiedKvTensorPool<B> {}
unsafe impl<B: aien_platform::UnifiedBuffer + Sync> Sync for UnifiedKvTensorPool<B> {}

impl<B: aien_platform::UnifiedBuffer> UnifiedKvTensorPool<B> {
    pub fn new(config: KvPoolConfig, buffer: B) -> Result<Self, String> {
        let layout = KvLayout::from_config(&config)?;
        if buffer.len() < layout.total_bytes {
            return Err(format!(
                "Buffer size {} is smaller than required KV tensor pool size {}",
                buffer.len(),
                layout.total_bytes
            ));
        }

        Ok(Self {
            config,
            layout,
            buffer,
        })
    }

    pub fn config(&self) -> &KvPoolConfig {
        &self.config
    }

    pub fn layout(&self) -> &KvLayout {
        &self.layout
    }

    pub fn layout_desc(&self) -> KvLayoutDesc {
        self.layout.to_desc()
    }

    pub fn total_bytes(&self) -> usize {
        self.layout.total_bytes
    }

    pub fn block_bytes(&self) -> usize {
        self.layout.block_stride_bytes
    }

    #[inline]
    pub fn element_strides(&self) -> (usize, usize, usize, usize) {
        let token_stride = self.config.num_kv_heads * self.config.head_dim;
        let kv_stride = self.config.block_size * token_stride;
        let layer_stride = 2 * kv_stride;
        let block_stride = self.config.num_layers * layer_stride;
        (token_stride, kv_stride, layer_stride, block_stride)
    }

    pub fn element_offset(
        &self,
        block_id: BlockId,
        layer_idx: usize,
        is_value: bool,
        token_in_block: usize,
    ) -> usize {
        self.layout
            .token_plane_offset(block_id, layer_idx, is_value, token_in_block)
    }

    pub fn base_ptr(&self) -> *const u8 {
        self.buffer.as_ptr()
    }

    pub fn base_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr()
    }

    pub fn device_address(&self) -> aien_platform::DeviceAddress {
        self.buffer.device_address()
    }

    pub fn block_device_address(&self, block_id: BlockId) -> aien_platform::DeviceAddress {
        let offset = self.layout.block_offset(block_id);
        aien_platform::DeviceAddress(self.buffer.device_address().0 + offset as u64)
    }

    pub fn block_region(&self, block_id: BlockId) -> BufferRegion {
        let offset = self.layout.block_offset(block_id);
        BufferRegion {
            address: aien_platform::DeviceAddress(self.buffer.device_address().0 + offset as u64),
            offset,
            len: self.layout.block_stride_bytes,
        }
    }

    pub fn buffer(&self) -> &B {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut B {
        &mut self.buffer
    }

    pub fn copy_block(&mut self, src_block: BlockId, dst_block: BlockId) {
        if src_block == dst_block {
            return;
        }
        let src_offset = self.layout.block_offset(src_block);
        let dst_offset = self.layout.block_offset(dst_block);
        let len = self.layout.block_stride_bytes;
        unsafe {
            let base = self.buffer.as_mut_ptr();
            let src = base.add(src_offset);
            let dst = base.add(dst_offset);
            std::ptr::copy(src, dst, len);
        }
    }

    pub fn zero_block(&mut self, block_id: BlockId) {
        let offset = self.layout.block_offset(block_id);
        let len = self.layout.block_stride_bytes;
        unsafe {
            let ptr = self.buffer.as_mut_ptr().add(offset);
            std::ptr::write_bytes(ptr, 0, len);
        }
    }

    pub fn write_token_kv(
        &mut self,
        block_id: BlockId,
        layer_idx: usize,
        token_in_block: usize,
        k: &[f32],
        v: &[f32],
    ) {
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        assert_eq!(k.len(), kv_dim, "K vector length must equal kv_dim");
        assert_eq!(v.len(), kv_dim, "V vector length must equal kv_dim");

        let k_offset = self
            .layout
            .token_plane_offset(block_id, layer_idx, false, token_in_block);
        let v_offset = self
            .layout
            .token_plane_offset(block_id, layer_idx, true, token_in_block);

        match self.config.dtype {
            KvDType::Fp32 => unsafe {
                let k_ptr = self.buffer.as_mut_ptr().add(k_offset) as *mut f32;
                let v_ptr = self.buffer.as_mut_ptr().add(v_offset) as *mut f32;
                std::ptr::copy_nonoverlapping(k.as_ptr(), k_ptr, kv_dim);
                std::ptr::copy_nonoverlapping(v.as_ptr(), v_ptr, kv_dim);
            },
            KvDType::Bf16 => unsafe {
                let k_ptr = self.buffer.as_mut_ptr().add(k_offset) as *mut u16;
                let v_ptr = self.buffer.as_mut_ptr().add(v_offset) as *mut u16;
                for i in 0..kv_dim {
                    *k_ptr.add(i) = f32_to_bf16_bits(k[i]);
                    *v_ptr.add(i) = f32_to_bf16_bits(v[i]);
                }
            },
            other => panic!("Unsupported KvDType for write_token_kv: {:?}", other),
        }
    }

    pub fn read_token_kv(
        &self,
        block_id: BlockId,
        layer_idx: usize,
        token_in_block: usize,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) {
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        assert_eq!(k_out.len(), kv_dim, "K_out length must equal kv_dim");
        assert_eq!(v_out.len(), kv_dim, "V_out length must equal kv_dim");

        let k_offset = self
            .layout
            .token_plane_offset(block_id, layer_idx, false, token_in_block);
        let v_offset = self
            .layout
            .token_plane_offset(block_id, layer_idx, true, token_in_block);

        match self.config.dtype {
            KvDType::Fp32 => unsafe {
                let k_ptr = self.buffer.as_ptr().add(k_offset) as *const f32;
                let v_ptr = self.buffer.as_ptr().add(v_offset) as *const f32;
                std::ptr::copy_nonoverlapping(k_ptr, k_out.as_mut_ptr(), kv_dim);
                std::ptr::copy_nonoverlapping(v_ptr, v_out.as_mut_ptr(), kv_dim);
            },
            KvDType::Bf16 => unsafe {
                let k_ptr = self.buffer.as_ptr().add(k_offset) as *const u16;
                let v_ptr = self.buffer.as_ptr().add(v_offset) as *const u16;
                for i in 0..kv_dim {
                    k_out[i] = bf16_bits_to_f32(*k_ptr.add(i));
                    v_out[i] = bf16_bits_to_f32(*v_ptr.add(i));
                }
            },
            other => panic!("Unsupported KvDType for read_token_kv: {:?}", other),
        }
    }

    pub fn gather_layer_kv(
        &self,
        block_table: &[BlockId],
        total_tokens: usize,
        layer_idx: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        let mut k_gathered = Vec::with_capacity(total_tokens * kv_dim);
        let mut v_gathered = Vec::with_capacity(total_tokens * kv_dim);

        let mut token_k = vec![0.0f32; kv_dim];
        let mut token_v = vec![0.0f32; kv_dim];

        let mut tokens_read = 0;
        for &blk in block_table {
            let tokens_in_this_block = (total_tokens - tokens_read).min(self.config.block_size);
            for tok_idx in 0..tokens_in_this_block {
                self.read_token_kv(blk, layer_idx, tok_idx, &mut token_k, &mut token_v);
                k_gathered.extend_from_slice(&token_k);
                v_gathered.extend_from_slice(&token_v);
            }
            tokens_read += tokens_in_this_block;
            if tokens_read >= total_tokens {
                break;
            }
        }

        (k_gathered, v_gathered)
    }
}

impl UnifiedKvTensorPool<HeapUnifiedBuffer> {
    pub fn allocate(config: KvPoolConfig) -> Result<Self, String> {
        let layout = KvLayout::from_config(&config)?;
        let buf = HeapUnifiedBuffer::allocate(layout.total_bytes, 256)?;
        Self::new(config, buf)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvBlock {
    pub block_id: BlockId,
    pub ref_count: usize,
    pub num_tokens: usize,
    pub is_shared: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockTable {
    pub sequence_id: u64,
    pub block_ids: Vec<BlockId>,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenReservation {
    pub seq_id: u64,
    pub block_id: BlockId,
    pub slot: usize,
    pub original_block_id: Option<BlockId>,
    pub was_cow: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KvMetrics {
    pub physical_pages: usize,
    pub logical_pages: usize,
    pub shared_pages: usize,
    pub private_pages: usize,
    pub cow_faults: usize,
    pub bytes_saved_vs_full_copy: usize,
    pub used_blocks: usize,
}

pub struct AienKvManager<B: aien_platform::UnifiedBuffer = HeapUnifiedBuffer> {
    total_blocks: usize,
    block_size: usize,
    free_blocks: Vec<BlockId>,
    blocks: Vec<KvBlock>,
    sequence_tables: HashMap<u64, BlockTable>,
    #[allow(dead_code)]
    #[allow(dead_code)]
    prefix_cache: HashMap<Vec<u32>, BlockId>,
    tensor_pool: Option<UnifiedKvTensorPool<B>>,
    pub cow_faults: usize,
}

impl AienKvManager<HeapUnifiedBuffer> {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        let mut free_blocks = Vec::with_capacity(total_blocks);
        let mut blocks = Vec::with_capacity(total_blocks);

        for id in (0..total_blocks).rev() {
            free_blocks.push(id);
        }

        for id in 0..total_blocks {
            blocks.push(KvBlock {
                block_id: id,
                ref_count: 0,
                num_tokens: 0,
                is_shared: false,
            });
        }

        Self {
            total_blocks,
            block_size,
            free_blocks,
            blocks,
            sequence_tables: HashMap::new(),
            prefix_cache: HashMap::new(),
            tensor_pool: None,
            cow_faults: 0,
        }
    }

    pub fn new_with_pool(
        total_blocks: usize,
        block_size: usize,
        config: KvPoolConfig,
    ) -> Result<Self, String> {
        let mut mgr = Self::new(total_blocks, block_size);
        mgr.attach_tensor_pool(config)?;
        Ok(mgr)
    }

    pub fn with_tensor_pool(
        total_blocks: usize,
        block_size: usize,
        config: KvPoolConfig,
    ) -> Result<Self, String> {
        Self::new_with_pool(total_blocks, block_size, config)
    }

    pub fn attach_tensor_pool(&mut self, config: KvPoolConfig) -> Result<(), String> {
        let pool = UnifiedKvTensorPool::allocate(config)?;
        self.tensor_pool = Some(pool);
        Ok(())
    }
}

impl<B: aien_platform::UnifiedBuffer> AienKvManager<B> {
    pub fn with_buffer(
        total_blocks: usize,
        block_size: usize,
        config: KvPoolConfig,
        buffer: B,
    ) -> Result<Self, String> {
        let mut free_blocks = Vec::with_capacity(total_blocks);
        let mut blocks = Vec::with_capacity(total_blocks);

        for id in (0..total_blocks).rev() {
            free_blocks.push(id);
        }

        for id in 0..total_blocks {
            blocks.push(KvBlock {
                block_id: id,
                ref_count: 0,
                num_tokens: 0,
                is_shared: false,
            });
        }

        let pool = UnifiedKvTensorPool::new(config, buffer)?;

        Ok(Self {
            total_blocks,
            block_size,
            free_blocks,
            blocks,
            sequence_tables: HashMap::new(),
            prefix_cache: HashMap::new(),
            tensor_pool: Some(pool),
            cow_faults: 0,
        })
    }

    pub fn attach_buffer(&mut self, config: KvPoolConfig, buffer: B) -> Result<(), String> {
        let pool = UnifiedKvTensorPool::new(config, buffer)?;
        self.tensor_pool = Some(pool);
        Ok(())
    }

    pub fn available_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn debug_active_blocks(&self) -> Vec<(usize, usize, bool)> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.ref_count > 0)
            .map(|(i, b)| (i, b.ref_count, b.is_shared))
            .collect()
    }

    pub fn active_sequence_count(&self) -> usize {
        self.sequence_tables.len()
    }

    pub fn fork_context(&mut self, parent_id: u64, child_id: u64) -> Result<Vec<BlockId>, String> {
        self.fork_sequence(parent_id, child_id)
    }

    pub fn release_branch(&mut self, seq_id: u64) {
        self.free_sequence(seq_id);
    }

    pub fn metrics(&self) -> KvMetrics {
        let physical_pages = self.blocks.iter().filter(|b| b.ref_count > 0).count();
        let logical_pages: usize = self
            .sequence_tables
            .values()
            .map(|t| t.block_ids.len())
            .sum();
        let shared_pages = self.blocks.iter().filter(|b| b.ref_count > 1).count();
        let private_pages = self.blocks.iter().filter(|b| b.ref_count == 1).count();
        let bytes_per_block = self
            .tensor_pool
            .as_ref()
            .map(|p| p.block_bytes())
            .unwrap_or(0);
        let bytes_saved_vs_full_copy =
            logical_pages.saturating_sub(physical_pages) * bytes_per_block;

        KvMetrics {
            physical_pages,
            logical_pages,
            shared_pages,
            private_pages,
            cow_faults: self.cow_faults,
            bytes_saved_vs_full_copy,
            used_blocks: physical_pages,
        }
    }

    pub fn pool(&self) -> Option<&UnifiedKvTensorPool<B>> {
        self.tensor_pool.as_ref()
    }

    pub fn pool_mut(&mut self) -> Option<&mut UnifiedKvTensorPool<B>> {
        self.tensor_pool.as_mut()
    }

    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn allocated_block_count(&self) -> usize {
        self.total_blocks - self.free_blocks.len()
    }

    pub fn allocate_block(&mut self) -> Result<BlockId, String> {
        let blk_id = self
            .free_blocks
            .pop()
            .ok_or_else(|| "Out of physical KV blocks".to_string())?;
        self.blocks[blk_id].ref_count = 1;
        self.blocks[blk_id].num_tokens = 0;
        self.blocks[blk_id].is_shared = false;

        if let Some(pool) = &mut self.tensor_pool {
            pool.zero_block(blk_id);
        }

        Ok(blk_id)
    }

    pub fn allocate_sequence(
        &mut self,
        seq_id: u64,
        prompt_tokens: &[u32],
    ) -> Result<Vec<BlockId>, String> {
        if let Some(table) = self.sequence_tables.get(&seq_id) {
            return Ok(table.block_ids.clone());
        }

        let num_tokens = prompt_tokens.len();
        let blocks_needed = num_tokens.div_ceil(self.block_size);

        if self.free_blocks.len() < blocks_needed {
            return Err(format!(
                "Insufficient blocks: need {}, have {}",
                blocks_needed,
                self.free_blocks.len()
            ));
        }

        let mut assigned = Vec::with_capacity(blocks_needed);
        let mut tokens_remaining = num_tokens;

        for _ in 0..blocks_needed {
            let blk = self.allocate_block()?;
            let tokens_in_blk = tokens_remaining.min(self.block_size);
            self.blocks[blk].num_tokens = tokens_in_blk;
            tokens_remaining -= tokens_in_blk;
            assigned.push(blk);
        }

        self.sequence_tables.insert(
            seq_id,
            BlockTable {
                sequence_id: seq_id,
                block_ids: assigned.clone(),
                total_tokens: num_tokens,
            },
        );

        Ok(assigned)
    }

    pub fn fork_sequence(&mut self, parent_id: u64, child_id: u64) -> Result<Vec<BlockId>, String> {
        let parent_table = self
            .sequence_tables
            .get(&parent_id)
            .ok_or_else(|| format!("Parent sequence {} not found", parent_id))?
            .clone();

        for &blk_id in &parent_table.block_ids {
            self.blocks[blk_id].ref_count += 1;
            self.blocks[blk_id].is_shared = true;
        }

        self.sequence_tables.insert(
            child_id,
            BlockTable {
                sequence_id: child_id,
                block_ids: parent_table.block_ids.clone(),
                total_tokens: parent_table.total_tokens,
            },
        );

        Ok(parent_table.block_ids)
    }

    pub fn append_token(&mut self, seq_id: u64) -> Result<BlockId, String> {
        let (block_id, _) = self.append_token_with_slot(seq_id)?;
        Ok(block_id)
    }

    pub fn append_token_with_slot(&mut self, seq_id: u64) -> Result<(BlockId, usize), String> {
        let table = self
            .sequence_tables
            .get_mut(&seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        if table.block_ids.is_empty() {
            let new_blk = self.allocate_block()?;
            let table = self.sequence_tables.get_mut(&seq_id).unwrap();
            table.block_ids.push(new_blk);
            table.total_tokens = 1;
            self.blocks[new_blk].num_tokens = 1;
            return Ok((new_blk, 0));
        }

        let last_blk_idx = table.block_ids.len() - 1;
        let last_blk_id = table.block_ids[last_blk_idx];
        let blk = &self.blocks[last_blk_id];

        if blk.num_tokens < self.block_size {
            let slot = blk.num_tokens;
            if blk.is_shared {
                let new_blk = self.allocate_block()?;
                if let Some(pool) = &mut self.tensor_pool {
                    pool.copy_block(last_blk_id, new_blk);
                }
                let old_blk = &mut self.blocks[last_blk_id];
                old_blk.ref_count -= 1;
                if old_blk.ref_count == 1 {
                    old_blk.is_shared = false;
                }

                let old_tokens = old_blk.num_tokens;
                self.blocks[new_blk].num_tokens = old_tokens + 1;
                self.blocks[new_blk].is_shared = false;
                self.cow_faults += 1;

                let table = self.sequence_tables.get_mut(&seq_id).unwrap();
                table.block_ids[last_blk_idx] = new_blk;
                table.total_tokens += 1;
                Ok((new_blk, slot))
            } else {
                self.blocks[last_blk_id].num_tokens += 1;
                table.total_tokens += 1;
                Ok((last_blk_id, slot))
            }
        } else {
            let new_blk = self.allocate_block()?;
            let table = self.sequence_tables.get_mut(&seq_id).unwrap();
            table.block_ids.push(new_blk);
            table.total_tokens += 1;
            self.blocks[new_blk].num_tokens = 1;
            Ok((new_blk, 0))
        }
    }

    pub fn reserve_token(&mut self, seq_id: u64) -> Result<TokenReservation, String> {
        let (block_id, slot) = self.append_token_with_slot(seq_id)?;
        Ok(TokenReservation {
            seq_id,
            block_id,
            slot,
            original_block_id: None,
            was_cow: false,
        })
    }

    pub fn commit(&mut self, _reservation: TokenReservation) -> Result<(), String> {
        Ok(())
    }

    pub fn rollback(&mut self, reservation: TokenReservation) -> Result<(), String> {
        let table = self
            .sequence_tables
            .get_mut(&reservation.seq_id)
            .ok_or_else(|| format!("Sequence {} not found", reservation.seq_id))?;
        table.total_tokens = table.total_tokens.saturating_sub(1);
        Ok(())
    }

    pub fn free_sequence(&mut self, seq_id: u64) {
        if let Some(table) = self.sequence_tables.remove(&seq_id) {
            for blk_id in table.block_ids {
                let blk = &mut self.blocks[blk_id];
                blk.ref_count -= 1;
                if blk.ref_count == 0 {
                    blk.is_shared = false;
                    blk.num_tokens = 0;
                    self.free_blocks.push(blk_id);
                } else if blk.ref_count == 1 {
                    blk.is_shared = false;
                }
            }
        }
    }

    pub fn get_block_table(&self, seq_id: u64) -> Option<&BlockTable> {
        self.sequence_tables.get(&seq_id)
    }

    pub fn get_block(&self, block_id: BlockId) -> Option<&KvBlock> {
        self.blocks.get(block_id)
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn tensor_pool(&self) -> Option<&UnifiedKvTensorPool<B>> {
        self.tensor_pool.as_ref()
    }

    pub fn tensor_pool_mut(&mut self) -> Option<&mut UnifiedKvTensorPool<B>> {
        self.tensor_pool.as_mut()
    }

    pub fn total_tokens(&self, seq_id: u64) -> Option<usize> {
        self.sequence_tables.get(&seq_id).map(|t| t.total_tokens)
    }

    pub fn write_token_kv(
        &mut self,
        seq_id: u64,
        token_pos: usize,
        layer_idx: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), String> {
        let table = self
            .sequence_tables
            .get(&seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        let block_idx = token_pos / self.block_size;
        let token_in_block = token_pos % self.block_size;

        if block_idx >= table.block_ids.len() {
            return Err(format!(
                "Token pos {} out of bounds for sequence {}",
                token_pos, seq_id
            ));
        }

        let physical_block = table.block_ids[block_idx];

        if let Some(pool) = &mut self.tensor_pool {
            pool.write_token_kv(physical_block, layer_idx, token_in_block, k, v);
            Ok(())
        } else {
            Err("No physical tensor pool attached".to_string())
        }
    }

    pub fn write_explicit_token_kv(
        &mut self,
        block_id: BlockId,
        layer_idx: usize,
        token_in_block: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), String> {
        if let Some(pool) = &mut self.tensor_pool {
            pool.write_token_kv(block_id, layer_idx, token_in_block, k, v);
            Ok(())
        } else {
            Err("No physical tensor pool attached".to_string())
        }
    }

    pub fn gather_sequence_layer_kv(
        &self,
        seq_id: u64,
        layer_idx: usize,
        k_out: &mut Vec<f32>,
        v_out: &mut Vec<f32>,
    ) -> Result<(), String> {
        let (k, v) = self.gather_layer_kv(seq_id, layer_idx)?;
        *k_out = k;
        *v_out = v;
        Ok(())
    }

    pub fn gather_layer_kv(
        &self,
        seq_id: u64,
        layer_idx: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let table = self
            .sequence_tables
            .get(&seq_id)
            .ok_or_else(|| format!("Sequence {} not found", seq_id))?;

        if let Some(pool) = &self.tensor_pool {
            Ok(pool.gather_layer_kv(&table.block_ids, table.total_tokens, layer_idx))
        } else {
            Err("No physical tensor pool attached".to_string())
        }
    }
}

pub type SharedKvManager<B = HeapUnifiedBuffer> = Arc<RwLock<AienKvManager<B>>>;

pub fn create_shared_kv_manager(
    total_blocks: usize,
    block_size: usize,
) -> SharedKvManager<HeapUnifiedBuffer> {
    Arc::new(RwLock::new(AienKvManager::new(total_blocks, block_size)))
}

pub fn create_shared_kv_manager_with_pool(
    total_blocks: usize,
    block_size: usize,
    config: KvPoolConfig,
) -> Result<SharedKvManager<HeapUnifiedBuffer>, String> {
    let mgr = AienKvManager::new_with_pool(total_blocks, block_size, config)?;
    Ok(Arc::new(RwLock::new(mgr)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle_offset(
        block: usize,
        layer: usize,
        is_value: bool,
        token: usize,
        head: usize,
        dim: usize,
        cfg: &KvPoolConfig,
    ) -> usize {
        let e = match cfg.dtype {
            KvDType::Bf16 | KvDType::Fp16 => 2usize,
            KvDType::Fp32 => 4usize,
            KvDType::Fp8 => 1usize,
            KvDType::Fp4 => 1usize,
        };

        let head_stride = cfg.head_dim * e;
        let token_stride = cfg.num_kv_heads * head_stride;
        let plane_stride = cfg.block_size * token_stride;
        let layer_stride = 2 * plane_stride;
        let block_stride = cfg.num_layers * layer_stride;

        block * block_stride
            + layer * layer_stride
            + usize::from(is_value) * plane_stride
            + token * token_stride
            + head * head_stride
            + dim * e
    }

    #[test]
    fn test_kv_layout_stride_invariants() {
        let cfg = KvPoolConfig {
            num_blocks: 8,
            block_size: 16,
            num_layers: 3,
            num_kv_heads: 2,
            head_dim: 64,
            dtype: KvDType::Bf16,
        };

        let layout = KvLayout::from_config(&cfg).unwrap();

        assert_eq!(layout.element_bytes, 2);
        assert_eq!(layout.head_stride_bytes, 128);
        assert_eq!(layout.token_stride_bytes, 256);
        assert_eq!(layout.kv_plane_stride_bytes, 4096);
        assert_eq!(layout.layer_stride_bytes, 8192);
        assert_eq!(layout.block_stride_bytes, 24576);
        assert_eq!(layout.total_bytes, 8 * 24576);

        let desc = layout.to_desc();
        assert_eq!(desc.head_stride_bytes, 128);
        assert_eq!(desc.token_stride_bytes, 256);
        assert_eq!(desc.kv_plane_stride_bytes, 4096);
        assert_eq!(desc.layer_stride_bytes, 8192);
        assert_eq!(desc.block_stride_bytes, 24576);
        assert_eq!(desc.pool_bytes, (8 * 24576) as u64);
    }

    #[test]
    fn test_kv_layout_independent_address_parity() {
        let cfg = KvPoolConfig::for_tinyllama(16, 16, KvDType::Bf16);
        let layout = KvLayout::from_config(&cfg).unwrap();

        let test_cases = [
            (0, 0, false, 0, 0, 0),
            (0, 0, true, 0, 0, 0),
            (0, 1, false, 0, 0, 0),
            (1, 0, false, 0, 0, 0),
            (7, 13, true, 4, 2, 31),
            (15, 21, true, 15, 3, 63),
        ];

        for &(blk, layer, is_val, tok, head, dim) in &test_cases {
            let expected = oracle_offset(blk, layer, is_val, tok, head, dim, &cfg);
            let actual = layout.element_offset(blk, layer, is_val, tok, head, dim);
            assert_eq!(
                expected, actual,
                "Offset mismatch at blk={} layer={} val={} tok={} head={} dim={}",
                blk, layer, is_val, tok, head, dim
            );
        }
    }

    #[test]
    fn test_bf16_pool_roundtrip() {
        let cfg = KvPoolConfig {
            num_blocks: 4,
            block_size: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            dtype: KvDType::Bf16,
        };

        let mut pool = UnifiedKvTensorPool::allocate(cfg.clone()).unwrap();
        let kv_dim = 2 * 8;

        let k_in: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.125).collect();
        let v_in: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 10.0) * 0.25).collect();

        pool.write_token_kv(1, 0, 3, &k_in, &v_in);

        let mut k_out = vec![0.0f32; kv_dim];
        let mut v_out = vec![0.0f32; kv_dim];
        pool.read_token_kv(1, 0, 3, &mut k_out, &mut v_out);

        for i in 0..kv_dim {
            assert!(
                (k_in[i] - k_out[i]).abs() < 1e-2,
                "K mismatch at {}: expected {}, got {}",
                i,
                k_in[i],
                k_out[i]
            );
            assert!(
                (v_in[i] - v_out[i]).abs() < 1e-2,
                "V mismatch at {}: expected {}, got {}",
                i,
                v_in[i],
                v_out[i]
            );
        }
    }

    #[test]
    fn test_physical_byte_pattern() {
        let cfg = KvPoolConfig {
            num_blocks: 4,
            block_size: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 4,
            dtype: KvDType::Bf16,
        };

        let mut pool = UnifiedKvTensorPool::allocate(cfg.clone()).unwrap();

        let coord_val = |b: usize, l: usize, is_v: bool, t: usize, h: usize, d: usize| -> f32 {
            b as f32 * 0.1
                + l as f32 * 0.01
                + if is_v { 0.005 } else { 0.0 }
                + t as f32 * 0.0001
                + h as f32 * 0.00001
                + d as f32 * 0.000001
        };

        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        for b in 0..2 {
            for l in 0..2 {
                for t in 0..4 {
                    let mut k = vec![0.0f32; kv_dim];
                    let mut v = vec![0.0f32; kv_dim];
                    for h in 0..cfg.num_kv_heads {
                        for d in 0..cfg.head_dim {
                            k[h * cfg.head_dim + d] = coord_val(b, l, false, t, h, d);
                            v[h * cfg.head_dim + d] = coord_val(b, l, true, t, h, d);
                        }
                    }
                    pool.write_token_kv(b, l, t, &k, &v);
                }
            }
        }

        let mut k_read = vec![0.0f32; kv_dim];
        let mut v_read = vec![0.0f32; kv_dim];
        pool.read_token_kv(0, 0, 0, &mut k_read, &mut v_read);
        assert_ne!(k_read[0], v_read[0], "K and V should not match");

        let mut k_l1 = vec![0.0f32; kv_dim];
        let mut v_l1 = vec![0.0f32; kv_dim];
        pool.read_token_kv(0, 1, 0, &mut k_l1, &mut v_l1);
        assert_ne!(k_read[0], k_l1[0], "Layer 0 and Layer 1 K should not match");
    }

    #[test]
    fn test_cow_manager_fork_and_divergence() {
        let cfg = KvPoolConfig::for_tinyllama(32, 16, KvDType::Bf16);
        let mut mgr = AienKvManager::new_with_pool(32, 16, cfg).unwrap();

        let prompt: Vec<u32> = (0..30).collect();
        let parent_blocks = mgr.allocate_sequence(100, &prompt).unwrap();
        assert_eq!(parent_blocks.len(), 2);

        let child1_blocks = mgr.fork_sequence(100, 201).unwrap();
        let child2_blocks = mgr.fork_sequence(100, 202).unwrap();

        assert_eq!(child1_blocks, parent_blocks);
        assert_eq!(child2_blocks, parent_blocks);

        // Append to child 1 -> triggers COW on block 1
        let (new_blk, _) = mgr.append_token_with_slot(201).unwrap();
        let child1_now = mgr.get_block_table(201).unwrap();

        assert_eq!(
            child1_now.block_ids[0], parent_blocks[0],
            "Prefix block 0 must remain shared"
        );
        assert_ne!(
            child1_now.block_ids[1], parent_blocks[1],
            "Tail block 1 must have diverged via COW"
        );
        assert_eq!(child1_now.block_ids[1], new_blk);

        // Parent and child 2 still share original block 1
        let parent_now = mgr.get_block_table(100).unwrap();
        let child2_now = mgr.get_block_table(202).unwrap();
        assert_eq!(parent_now.block_ids[1], parent_blocks[1]);
        assert_eq!(child2_now.block_ids[1], parent_blocks[1]);
    }
}
