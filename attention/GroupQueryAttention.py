"""
分组查询注意力（Grouped Query Attention, GQA）

一种介于多头注意力（MHA）和多查询注意力（MQA）之间的注意力机制。
通过让多个查询头共享同一组 KV 头，在性能和效率之间取得平衡。

参考论文: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GroupQueryAttention(nn.Module):
    """
    分组查询注意力模块

    在 GQA 中：
    - Q 有 num_heads 个头
    - K 和 V 只有 num_kv_heads 个头（num_kv_heads < num_heads）
    - 多个 Q 头共享同一个 KV 头（通过 repeat_kv 实现）

    优势：
    - 比 MHA 更少的 KV 缓存，提高推理效率
    - 比 MQA 更好的性能，保持模型质量

    Args:
        model_dim: 模型隐藏维度
        num_heads: 查询头数
        num_kv_heads: 键值头数（必须能整除 num_heads）
        dropout_p: Dropout 概率，默认 0.0
    """

    def __init__(self, model_dim, num_heads, num_kv_heads, dropout_p=0.0):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.head_dim = model_dim // num_heads
        self.num_rep = num_heads // num_kv_heads  # 每个 KV 头被复制的次数

        # Q 投影层：输出 num_heads 个头
        self.w_q = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)

        # K, V 投影层：输出 num_kv_heads 个头
        self.w_k = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)

        # 输出投影层
        self.w_o = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout_p)

    def repeat_kv(self, x, n_rep):
        """
        复制 KV 头以匹配 Q 头数

        将 num_kv_heads 个头复制扩展为 num_heads 个头。

        Args:
            x: 输入张量 [batch_size, num_kv_heads, seq_len, head_dim]
            n_rep: 复制次数（= num_heads / num_kv_heads）

        Returns:
            扩展后的张量 [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_kv_heads, seq_len, head_dim = x.shape

        if n_rep == 1:
            return x

        # 步骤1: 在第二维后新增一个维度
        # [batch_size, num_kv_heads, 1, seq_len, head_dim]
        x = x[:, :, None, :, :]

        # 步骤2: 在新维度上扩展 n_rep 次
        # [batch_size, num_kv_heads, n_rep, seq_len, head_dim]
        x = x.expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)

        # 步骤3: 展平 num_kv_heads 和 n_rep 维度
        # [batch_size, num_kv_heads * n_rep, seq_len, head_dim] = [batch_size, num_heads, seq_len, head_dim]
        x = x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

        return x

    def forward(self, x, mask=None):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, model_dim]
            mask: 注意力掩码 [batch_size, 1, seq_len, seq_len] 或 [1, 1, seq_len, seq_len]

        Returns:
            output: 注意力输出 [batch_size, seq_len, model_dim]
        """
        batch_size, seq_len, _ = x.shape

        # ========== 线性变换得到 Q, K, V ==========
        # q: [batch_size, seq_len, num_heads * head_dim]
        q = self.w_q(x)
        # k: [batch_size, seq_len, num_kv_heads * head_dim]
        k = self.w_k(x)
        # v: [batch_size, seq_len, num_kv_heads * head_dim]
        v = self.w_v(x)

        # ========== 分头处理 ==========
        # q: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # k: [batch_size, seq_len, num_kv_heads, head_dim] -> [batch_size, num_kv_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # v: [batch_size, seq_len, num_kv_heads, head_dim] -> [batch_size, num_kv_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ========== 复制 K, V 以匹配 Q 的头数 ==========
        # k: [batch_size, num_heads, seq_len, head_dim]
        k = self.repeat_kv(k, self.num_rep)
        # v: [batch_size, num_heads, seq_len, head_dim]
        v = self.repeat_kv(v, self.num_rep)

        # ========== 计算注意力得分 ==========
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ========== 计算上下文向量 ==========
        # context: [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)

        # ========== 拼接多头 ==========
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        context = context.transpose(1, 2)
        context = context.contiguous()

        # [batch_size, seq_len, num_heads * head_dim] = [batch_size, seq_len, model_dim]
        output = context.view(batch_size, seq_len, self.model_dim)
        output = self.w_o(output)

        return output
