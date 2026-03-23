"""
旋转位置编码（Rotary Position Embedding, RoPE）

一种将位置信息编码到注意力机制中的方法，通过旋转向量来表示相对位置。
具有良好的外推性和位置感知能力。

参考论文: RoFormer: Enhanced Transformer with Rotary Position Embedding
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块

    通过将查询和键向量按照位置进行旋转，使注意力分数包含相对位置信息。
    公式: f(x, m) = x * cos(m*θ) + rotate_half(x) * sin(m*θ)

    Args:
        head_dim: 旋转编码的维度（通常是 attention head 的维度）
        max_seq_len: 支持的最大序列长度
        theta: 旋转角度的基数，默认 10000.0
    """

    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算 cos 和 sin 值，避免重复计算
        cos, sin = self.precompute_freqs(head_dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def precompute_freqs(self, head_dim, max_seq_len, theta):
        """
        预计算旋转位置编码的频率

        Args:
            head_dim: 编码维度
            max_seq_len: 最大序列长度
            theta: 旋转基数

        Returns:
            cos: 余弦值 [max_seq_len, head_dim]
            sin: 正弦值 [max_seq_len, head_dim]
        """
        # 计算逆频率: 1 / (theta^(2i/d))
        # inv_freqs: [head_dim/2]
        inv_freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # 位置索引
        # t: [max_seq_len]
        t = torch.arange(max_seq_len, device=inv_freqs.device, dtype=torch.float32)

        # 计算角度矩阵: outer(t, inv_freqs)
        # angles: [max_seq_len, head_dim/2]
        angles = torch.outer(t, inv_freqs)

        # 将角度复制一份以匹配完整维度
        # angles: [max_seq_len, head_dim]
        angles = torch.cat((angles, angles), dim=-1)

        return angles.cos(), angles.sin()

    def forward(self, xq, xk):
        """
        对查询和键应用旋转位置编码

        Args:
            xq: 查询张量 [batch_size, seq_len, num_heads, head_dim]
            xk: 键张量 [batch_size, seq_len, num_heads, head_dim]

        Returns:
            xq_rotated: 旋转后的查询 [batch_size, seq_len, num_heads, head_dim]
            xk_rotated: 旋转后的键 [batch_size, seq_len, num_heads, head_dim]
        """
        seq_len = xq.size(1)

        # 获取当前位置的 cos 和 sin 值
        # cos, sin: [1, seq_len, 1, head_dim]
        cos = self.cos[:seq_len].view(1, seq_len, 1, self.head_dim)
        sin = self.sin[:seq_len].view(1, seq_len, 1, self.head_dim)

        def rotate_half(x):
            """
            将张量分成两半并旋转

            Args:
                x: 输入张量 [..., head_dim]

            Returns:
                旋转后的张量 [..., head_dim]
            """
            # x1, x2: [..., head_dim/2]
            x1, x2 = torch.chunk(x, 2, dim=-1)
            # 拼接为 [-x2, x1]: [..., head_dim]
            return torch.cat((-x2, x1), dim=-1)

        # 应用旋转位置编码
        # 公式: x * cos + rotate_half(x) * sin
        #
        # 详细推导:
        # 设 x = [x1, x2]，则 rotate_half(x) = [-x2, x1]
        # x * cos = [x1*cos, x2*cos]
        # rotate_half(x) * sin = [-x2*sin, x1*sin]
        # 结果 = [x1*cos - x2*sin, x2*cos + x1*sin]
        #
        # 等价于旋转矩阵: [cos, sin; -sin, cos] @ [x1; x2] = [x1'; x2']

        xq_rotated = (xq * cos) + (rotate_half(xq) * sin)
        xk_rotated = (xk * cos) + (rotate_half(xk) * sin)

        return xq_rotated, xk_rotated
