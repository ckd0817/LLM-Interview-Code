"""
多头潜在注意力（Multi-Head Latent Attention, MLA）

DeepSeek 提出的高效注意力机制，通过低秩压缩 KV 缓存来减少显存占用。
核心思想：将 KV 压缩到潜在空间，推理时只需缓存潜在向量。

参考论文: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from RotaryEmbedding import RotaryEmbedding


class MultiLatentAttention(nn.Module):
    """
    多头潜在注意力模块

    通过低秩投影将 Q 和 KV 压缩到潜在空间，显著减少 KV 缓存的显存占用。
    同时结合 RoPE 位置编码保持位置感知能力。

    Args:
        d_model: 模型隐藏维度
        num_heads: 注意力头数
        d_head: 每个注意力头的维度
        d_latent: 潜在空间维度（压缩后的维度）
        d_rope: RoPE 旋转位置编码维度
        dropout_p: Dropout 概率
    """

    def __init__(self, d_model, num_heads, d_head, d_latent, d_rope, dropout_p):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_latent = d_latent
        self.d_rope = d_rope

        # 1. KV 压缩投影：d_model -> d_latent（下采样）-> num_heads * (d_head + d_rope + d_head)（上采样）
        self.kv_down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.kv_up_proj = nn.Linear(d_latent, num_heads * (d_head + d_rope + d_head), bias=False)

        # 2. Q 压缩投影：d_model -> d_latent（下采样）-> num_heads * (d_head + d_rope)（上采样）
        self.q_down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.q_up_proj = nn.Linear(d_latent, num_heads * (d_head + d_rope), bias=False)

        # 3. 输出投影
        self.o_proj = nn.Linear(num_heads * d_head, d_model, bias=False)

        # 4. RoPE 旋转位置编码
        self.rope = RotaryEmbedding(d_rope)

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码，用于屏蔽某些位置 [batch_size, num_heads, seq_len, seq_len] 或 [1, 1, seq_len, seq_len]

        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # ========== KV 投影 ==========
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_latent]
        kv_latent = self.kv_down_proj(x)
        # [batch_size, seq_len, d_latent] -> [batch_size, seq_len, num_heads * (d_head + d_rope + d_head)]
        kv_full = self.kv_up_proj(kv_latent)
        # [batch_size, seq_len, num_heads * (d_head + d_rope + d_head)] -> [batch_size, seq_len, num_heads, d_head + d_rope + d_head]
        kv_full = kv_full.view(batch_size, seq_len, self.num_heads, -1)

        # 分离内容部分和 RoPE 部分
        # k_content: [batch_size, seq_len, num_heads, d_head]
        # k_rope: [batch_size, seq_len, num_heads, d_rope]
        # v_content: [batch_size, seq_len, num_heads, d_head]
        k_content, k_rope, v_content = torch.split(kv_full, [self.d_head, self.d_rope, self.d_head], dim=-1)

        # ========== Q 投影 ==========
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_latent]
        q_latent = self.q_down_proj(x)
        # [batch_size, seq_len, d_latent] -> [batch_size, seq_len, num_heads * (d_head + d_rope)]
        q_full = self.q_up_proj(q_latent)
        # [batch_size, seq_len, num_heads * (d_head + d_rope)] -> [batch_size, seq_len, num_heads, d_head + d_rope]
        q_full = q_full.view(batch_size, seq_len, self.num_heads, self.d_head + self.d_rope)

        # 分离内容部分和 RoPE 部分
        q_content, q_rope = torch.split(q_full, [self.d_head, self.d_rope], dim=-1)

        # ========== 应用 RoPE ==========
        # 对 q_rope 和 k_rope 应用旋转位置编码
        q_rope, k_rope = self.rope(q_rope, k_rope)

        # ========== 合并内容和 RoPE 部分 ==========
        # [batch_size, seq_len, num_heads, d_head + d_rope]
        q = torch.cat([q_content, q_rope], dim=-1)
        q.transpose_(1, 2)  # [batch_size, num_heads, seq_len, d_head + d_rope]

        # [batch_size, seq_len, num_heads, d_head + d_rope]
        k = torch.cat([k_content, k_rope], dim=-1)
        k.transpose_(1, 2)  # [batch_size, num_heads, seq_len, d_head + d_rope]

        # [batch_size, num_heads, seq_len, d_head]
        v = v_content.transpose(1, 2)

        # ========== 缩放点积注意力 ==========
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head + self.d_rope)

        # 应用注意力掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)

        # context: [batch_size, num_heads, seq_len, d_head]
        context = torch.matmul(attn_weights, v)

        # [batch_size, num_heads, seq_len, d_head] -> [batch_size, seq_len, num_heads * d_head]
        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_head)

        # [batch_size, seq_len, num_heads * d_head] -> [batch_size, seq_len, d_model]
        output = self.o_proj(output)

        return output
