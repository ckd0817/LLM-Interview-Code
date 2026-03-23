"""
多头注意力（Multi-Head Attention）

Transformer 的核心组件，通过并行运行多个注意力头来捕捉不同子空间的特征。
每个头独立计算注意力，最后将结果拼接并通过线性层融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块

    支持自注意力（Self-Attention）和交叉注意力（Cross-Attention）：
    - 自注意力：Q = K = V = x_query
    - 交叉注意力：Q = x_query, K = V = x_context

    Args:
        model_dim: 模型隐藏维度
        num_heads: 注意力头数
        dropout_p: Dropout 概率，默认 0.1
    """

    def __init__(self, model_dim, num_heads, dropout_p=0.1):
        super().__init__()

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads  # 每个头的维度

        # Q, K, V 投影层
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)

        # 输出投影层
        self.w_o = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x_query, x_context, mask=None):
        """
        前向传播

        Args:
            x_query: 查询输入 [batch_size, seq_len_q, model_dim]
            x_context: 上下文输入（用于生成 K 和 V）[batch_size, seq_len_k, model_dim]
                       如果为 None，则使用 x_query（自注意力）
            mask: 注意力掩码 [batch_size, 1, seq_len_q, seq_len_k] 或 [1, 1, seq_len_q, seq_len_k]

        Returns:
            output: 注意力输出 [batch_size, seq_len_q, model_dim]
        """
        batch_size = x_query.size(0)

        # ========== 线性投影 ==========
        # 自注意力: q = k = v = x_query
        # 交叉注意力: q = x_query, k = v = x_context
        q = self.w_q(x_query)

        if x_context is not None:
            k = self.w_k(x_context)
            v = self.w_v(x_context)
        else:
            k = self.w_k(x_query)
            v = self.w_v(x_query)

        # ========== 分头处理 ==========
        # [batch_size, seq_len, model_dim] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # ========== 缩放点积注意力 ==========
        # scores: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用注意力掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # attn_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # context: [batch_size, num_heads, seq_len_q, head_dim]
        context = torch.matmul(attn_weights, v)

        # ========== 合并多头 ==========
        # [batch_size, num_heads, seq_len_q, head_dim] -> [batch_size, seq_len_q, num_heads, head_dim] -> [batch_size, seq_len_q, model_dim]
        context = context.transpose(1, 2)
        context = context.contiguous()
        output = context.view(batch_size, -1, self.model_dim)

        # 输出投影
        output = self.w_o(output)

        return output


if __name__ == "__main__":
    # batch_size=2, seq_len=10, model_dim=64, num_heads=8
    x = torch.randn(2, 10, 64)
    mha = MultiHeadAttention(model_dim=64, num_heads=8)
    out = mha(x, x)  # Self-Attention: x_query=x, x_context=x
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  # 应该还是 (2, 10, 64)
