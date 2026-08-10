"""
Flash Attention（教学版）

FlashAttention 通过双层分块（tiling）+ online softmax，在不物化 n x n
注意力矩阵的前提下计算出与标准注意力完全一致的结果。
真实实现依赖 CUDA kernel fusion，将中间结果全程保留在 SRAM/寄存器中，
从而将 HBM 访问量从 O(n^2) 降到 O(n^2 * d / SRAM)。

参考论文:
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

注意：这是教学版模拟。PyTorch 中每个算子仍是独立 kernel，中间结果仍
往返 HBM，因此本实现只能演示"峰值显存下降"与算法正确性，
无法体现真实 FlashAttention 的 IO 优化。
"""

import math
import torch
import torch.nn as nn


def flash_attention(Q, K, V, block_q=64, block_k=64, causal=False):
    """
    教学版 FlashAttention 前向：双层分块 + online softmax

    与"只切 Q 的 query-chunking"的本质区别：
    1. Q、KV 两个维度都分块，任意时刻只存在 block_q x block_k 的 score tile
    2. softmax 分母需要整行 scores，KV 分块后每次只能看到一行的一段，
       因此必须用 online softmax 增量维护

    循环不变量（每处理完一个 KV block 后保持）：
    - m:   已见部分的行最大值
    - l:   sum(exp(S - m))，已见部分的 softmax 分母
    - acc: sum(exp(S - m) @ V)，未归一化的输出

    Args:
        Q, K, V: [batch, num_heads, seq_len, head_dim]
        block_q: Q 方向分块大小
        block_k: KV 方向分块大小
        causal:  是否使用因果掩码

    Returns:
        输出张量 [batch, num_heads, seq_len, head_dim]，与标准注意力数值一致
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    O = torch.zeros_like(Q)

    # 外层：遍历 Q block（真实 kernel 中一个 block 对应一个 CTA，数据驻留 SRAM）
    for i in range(0, N, block_q):
        q = Q[:, :, i:i + block_q]                      # [B, H, Bq, d]
        Bq = q.shape[2]

        # online softmax 的三个状态量，全程驻留 "SRAM"
        m = q.new_full((B, H, Bq, 1), -float("inf"))    # 运行时行最大值
        l = torch.zeros_like(m)                         # 运行时 softmax 分母
        acc = torch.zeros_like(q)                       # 未归一化的输出累加器

        # 内层：遍历 KV block
        for j in range(0, N, block_k):
            if causal and j >= i + Bq:                  # 整块在对角线上方，直接跳过
                break
            k = K[:, :, j:j + block_k]                  # [B, H, Bk, d]
            v = V[:, :, j:j + block_k]

            S = q @ k.transpose(-2, -1) * scale         # [B, H, Bq, Bk]

            if causal:
                row = torch.arange(i, i + Bq, device=Q.device)[:, None]
                col = torch.arange(j, j + k.shape[2], device=Q.device)[None, :]
                S = S.masked_fill(row < col, -float("inf"))

            # ---- online softmax 核心：三步更新 ----
            m_new = torch.maximum(m, S.amax(dim=-1, keepdim=True))
            P = torch.exp(S - m_new)                    # 当前块的未归一化概率
            alpha = torch.exp(m - m_new)                # 旧最大值失效带来的修正因子
            l = l * alpha + P.sum(dim=-1, keepdim=True)
            acc = acc * alpha + P @ v
            m = m_new

        # 最后统一归一化（FA2 的做法；FA1 是每个内层迭代都归一化一次）
        O[:, :, i:i + block_q] = acc / l
    return O


class FlashAttention(nn.Module):
    """
    基于教学版 flash_attention 的多头注意力模块

    Args:
        model_dim:  模型隐藏维度
        num_heads:  注意力头数
        block_size: 分块大小（Q 和 KV 方向共用）
        causal:     是否使用因果掩码
    """

    def __init__(self, model_dim=512, num_heads=8, block_size=64, causal=False):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.block_size = block_size
        self.causal = causal

        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_o = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch, seq_len, model_dim]

        Returns:
            输出张量 [batch, seq_len, model_dim]
        """
        B, N, _ = x.shape
        shape = (B, N, self.num_heads, self.head_dim)
        Q = self.W_q(x).view(shape).transpose(1, 2)     # [B, H, N, head_dim]
        K = self.W_k(x).view(shape).transpose(1, 2)
        V = self.W_v(x).view(shape).transpose(1, 2)

        O = flash_attention(Q, K, V, self.block_size, self.block_size, self.causal)

        O = O.transpose(1, 2).reshape(B, N, -1)         # [B, N, model_dim]
        return self.W_o(O)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 256, 512)

    attn = FlashAttention(model_dim=512, num_heads=8, block_size=64, causal=True)
    out = attn(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")

    # 与标准注意力对比，验证数值一致
    Q = attn.W_q(x).view(2, 256, 8, 64).transpose(1, 2)
    K = attn.W_k(x).view(2, 256, 8, 64).transpose(1, 2)
    V = attn.W_v(x).view(2, 256, 8, 64).transpose(1, 2)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    flash = flash_attention(Q, K, V, causal=True)
    print(f"与标准注意力最大误差: {(flash - ref).abs().max().item():.2e}")
