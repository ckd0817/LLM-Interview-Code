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
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from position.RotaryEmbedding import RotaryEmbedding


class MultiLatentAttention(nn.Module):
    """
    多头潜在注意力模块

    通过低秩投影将 Q 和 KV 压缩到潜在空间，显著减少 KV 缓存的显存占用。
    同时结合 RoPE 位置编码保持位置感知能力。

    Args:
        model_dim: 模型隐藏维度
        num_heads: 注意力头数
        head_dim: 每个注意力头的维度
        latent_dim: 潜在空间维度（压缩后的维度）
        rope_dim: RoPE 旋转位置编码维度
        dropout_p: Dropout 概率，默认 0.0
    """

    def __init__(self, model_dim, num_heads, head_dim, latent_dim, rope_dim, dropout_p=0.0):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim

        # 1. KV 压缩投影：model_dim -> latent_dim（下采样）-> num_heads * (head_dim + rope_dim + head_dim)（上采样）
        self.kv_down_proj = nn.Linear(model_dim, latent_dim, bias=False)
        self.kv_up_proj = nn.Linear(latent_dim, num_heads * (head_dim + rope_dim + head_dim), bias=False)

        # 2. Q 压缩投影：model_dim -> latent_dim（下采样）-> num_heads * (head_dim + rope_dim)（上采样）
        self.q_down_proj = nn.Linear(model_dim, latent_dim, bias=False)
        self.q_up_proj = nn.Linear(latent_dim, num_heads * (head_dim + rope_dim), bias=False)

        # 3. 输出投影
        self.o_proj = nn.Linear(num_heads * head_dim, model_dim, bias=False)

        self.dropout = nn.Dropout(dropout_p)

        # 4. RoPE 旋转位置编码
        self.rope = RotaryEmbedding(head_dim=rope_dim)

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, model_dim]
            mask: 注意力掩码，用于屏蔽某些位置 [batch_size, num_heads, seq_len, seq_len] 或 [1, 1, seq_len, seq_len]

        Returns:
            输出张量 [batch_size, seq_len, model_dim]
        """
        batch_size, seq_len, _ = x.size()

        # ========== KV 投影 ==========
        # [batch_size, seq_len, model_dim] -> [batch_size, seq_len, latent_dim]
        kv_latent = self.kv_down_proj(x)
        # [batch_size, seq_len, latent_dim] -> [batch_size, seq_len, num_heads * (head_dim + rope_dim + head_dim)]
        kv_full = self.kv_up_proj(kv_latent)
        # [batch_size, seq_len, num_heads * (head_dim + rope_dim + head_dim)] -> [batch_size, seq_len, num_heads, head_dim + rope_dim + head_dim]
        kv_full = kv_full.view(batch_size, seq_len, self.num_heads, -1)

        # 分离内容部分和 RoPE 部分
        # k_content: [batch_size, seq_len, num_heads, head_dim]
        # k_rope: [batch_size, seq_len, num_heads, rope_dim]
        # v_content: [batch_size, seq_len, num_heads, head_dim]
        k_content, k_rope, v_content = torch.split(kv_full, [self.head_dim, self.rope_dim, self.head_dim], dim=-1)

        # ========== Q 投影 ==========
        # [batch_size, seq_len, model_dim] -> [batch_size, seq_len, latent_dim]
        q_latent = self.q_down_proj(x)
        # [batch_size, seq_len, latent_dim] -> [batch_size, seq_len, num_heads * (head_dim + rope_dim)]
        q_full = self.q_up_proj(q_latent)
        # [batch_size, seq_len, num_heads * (head_dim + rope_dim)] -> [batch_size, seq_len, num_heads, head_dim + rope_dim]
        q_full = q_full.view(batch_size, seq_len, self.num_heads, self.head_dim + self.rope_dim)

        # 分离内容部分和 RoPE 部分
        q_content, q_rope = torch.split(q_full, [self.head_dim, self.rope_dim], dim=-1)

        # ========== 应用 RoPE ==========
        # 对 q_rope 和 k_rope 应用旋转位置编码
        q_rope, k_rope = self.rope(q_rope, k_rope)

        # ========== 合并内容和 RoPE 部分 ==========
        # [batch_size, seq_len, num_heads, head_dim + rope_dim]
        q = torch.cat([q_content, q_rope], dim=-1)
        q.transpose_(1, 2)  # [batch_size, num_heads, seq_len, head_dim + rope_dim]

        # [batch_size, seq_len, num_heads, head_dim + rope_dim]
        k = torch.cat([k_content, k_rope], dim=-1)
        k.transpose_(1, 2)  # [batch_size, num_heads, seq_len, head_dim + rope_dim]

        # [batch_size, num_heads, seq_len, head_dim]
        v = v_content.transpose(1, 2)

        # ========== 缩放点积注意力 ==========
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim + self.rope_dim)

        # 应用注意力掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # context: [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)

        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads * head_dim]
        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        # [batch_size, seq_len, num_heads * head_dim] -> [batch_size, seq_len, model_dim]
        output = self.o_proj(output)

        return output
