"""
缩放点积注意力（Scaled Dot-Product Attention）

Transformer 注意力机制的核心计算模块。
公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力模块

    计算查询和键的点积，除以缩放因子后应用 softmax 得到注意力权重，
    最后用注意力权重对值进行加权求和。

    Args:
        dropout_p: Dropout 概率，默认 0.0
    """

    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        """
        前向传播

        Args:
            q: 查询张量 [batch_size, num_heads, seq_len_q, head_dim]
            k: 键张量 [batch_size, num_heads, seq_len_k, head_dim]
            v: 值张量 [batch_size, num_heads, seq_len_k, head_dim]
            mask: 注意力掩码，0 表示被屏蔽的位置 [batch_size, 1, seq_len_q, seq_len_k] 或 [1, 1, seq_len_q, seq_len_k]

        Returns:
            output: 注意力输出 [batch_size, num_heads, seq_len_q, head_dim]
            attn_weights: 注意力权重 [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        head_dim = q.size(-1)

        # 步骤1: 计算注意力得分（缩放点积）
        # scores: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # 步骤2: 应用注意力掩码
        if mask is not None:
            # 被屏蔽的位置设为 -1e9，softmax 后接近 0
            scores = scores.masked_fill(mask == 0, -1e9)

        # 步骤3: Softmax 归一化得到注意力权重
        # attn_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # 步骤4: 加权求和
        # output: [batch_size, num_heads, seq_len_q, head_dim]
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入: batch_size=2, num_heads=4, seq_len=8, head_dim=64
    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 64
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 构造一个 Causal Mask (下三角矩阵)，模拟 GPT 生成过程
    # shape: (seq_len, seq_len), 上三角为 0, 下三角为 1
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

    attention_layer = ScaledDotProductAttention()
    output, weights = attention_layer(q, k, v, mask=causal_mask)

    print(f"Output shape: {output.shape}")  # Should be (2, 4, 8, 64)
    print(f"Weights shape: {weights.shape}")  # Should be (2, 4, 8, 8)

    # 验证 Mask 是否生效：查看第一个样本第一个头的第一行
    # 理论上只有第1个位置有值，后面全是0
    print("\nCheck Causal Masking (Row 0 should only attend to Col 0):")
    print(weights[0, 0, 0, :])
