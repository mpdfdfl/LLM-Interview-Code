作者：tachikoma
链接：https://zhuanlan.zhihu.com/p/2022092403721479678
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

class PageAttentionKVCache:
    """
    vLLM的核心创新：将KV Cache分页管理，解决动态长度内存碎片问题
    
    传统KV Cache问题：不同序列长度不同，预分配max_seq_len浪费严重
    PageAttention：将KV分成固定大小的block（page），按需分配
    
    面试常考：这是系统优化的经典案例，展现对LLM Serving的理解
    """
    def __init__(self, num_layers: int, num_heads: int, d_k: int, 
                 block_size: int = 16, num_blocks: int = 1000):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_k = d_k
        self.block_size = block_size
        
        # 全局block池，所有序列共享
        # shape: (num_blocks, num_heads, block_size, d_k)
        self.k_blocks = torch.zeros(num_blocks, num_heads, block_size, d_k)
        self.v_blocks = torch.zeros(num_blocks, num_heads, block_size, d_k)
        
        # block分配表：记录哪些block被占用
        self.block_table = {}  # seq_id -> list of block_indices
        
        # 空闲block列表
        self.free_blocks = list(range(num_blocks))
    
    def allocate(self, seq_id: int, num_tokens: int):
        """
        为新序列分配block
        如果序列增长，动态追加block
        """
        num_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if seq_id not in self.block_table:
            # 新序列，分配新block
            allocated = self.free_blocks[:num_needed]
            self.free_blocks = self.free_blocks[num_needed:]
            self.block_table[seq_id] = allocated
        else:
            # 已有序列，检查是否需要追加
            current_blocks = len(self.block_table[seq_id])
            if num_needed > current_blocks:
                additional = self.free_blocks[:num_needed - current_blocks]
                self.free_blocks = self.free_blocks[num_needed - current_blocks:]
                self.block_table[seq_id].extend(additional)
        
        return self.block_table[seq_id]
    
    def get_kv(self, seq_id: int, layer_idx: int):
        """
        根据block table gather KV
        
        面试难点：如何将分散的blocks拼接成连续的tensor用于attention
        """
        blocks = self.block_table[seq_id]
        
        # Gather blocks: list of (num_heads, block_size, d_k) -> (num_blocks, heads, block_size, d_k)
        k_gathered = torch.stack([self.k_blocks[b] for b in blocks])
        v_gathered = torch.stack([self.v_blocks[b] for b in blocks])
        
        # 展平为连续序列: (num_blocks, heads, block_size, d_k) -> (heads, num_blocks*block_size, d_k)
        seq_len = len(blocks) * self.block_size
        k_continuous = k_gathered.transpose(0, 1).reshape(self.num_heads, seq_len, self.d_k)
        v_continuous = v_gathered.transpose(0, 1).reshape(self.num_heads, seq_len, self.d_k)
        
        return k_continuous, v_continuous
    
    def append_kv(self, seq_id: int, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """
        追加新token的KV到对应block
        
        new_k, new_v: (num_heads, new_seq_len, d_k)
        """
        # 找到当前序列的最后一个block和偏移
        blocks = self.block_table[seq_id]
        current_len = self.get_current_length(seq_id)
        
        start_block_idx = current_len // self.block_size
        start_offset = current_len % self.block_size
        
        # 填充逻辑...
        # 面试简化版：假设总是对齐block边界
        block_idx = blocks[start_block_idx]
        offset = start_offset
        
        # 写入（实际需要考虑跨block情况）
        seq_len = new_k.size(1)
        self.k_blocks[block_idx, :, offset:offset+seq_len, :] = new_k
        self.v_blocks[block_idx, :, offset:offset+seq_len, :] = new_v
    
    def get_current_length(self, seq_id: int):
        """获取序列当前长度（需要额外维护，简化版）"""
        return len(self.block_table[seq_id]) * self.block_size  # 简化假设