"""
RMS 归一化（Root Mean Square Layer Normalization）

LayerNorm 的简化变体，去掉了均值中心化步骤，只使用 RMS 进行归一化。
计算量更小，在某些场景下效果相当或更好。

参考论文: Root Mean Square Layer Normalization
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMS 归一化模块

    公式: RMSNorm(x) = x / RMS(x) * gamma
    其中 RMS(x) = sqrt(mean(x^2) + eps)

    相比 LayerNorm，RMSNorm 不计算均值，计算量更小。

    Args:
        model_dim: 归一化的维度（通常是 model_dim）
        eps: 数值稳定性常数，防止除零，默认 1e-8
    """

    def __init__(self, model_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_dim))  # 可学习的缩放参数

    def _norm(self, x):
        """
        RMS 归一化核心计算

        Args:
            x: 输入张量 [batch_size, seq_len, model_dim]

        Returns:
            归一化后的张量 [batch_size, seq_len, model_dim]
        """
        # 计算均方值
        # mean_square: [batch_size, seq_len, 1]
        mean_square = x.float().pow(2).mean(-1, keepdim=True)

        # 计算 1/sqrt(x)，即 RMS 的倒数
        # rsqrt: [batch_size, seq_len, 1]
        rsqrt = torch.rsqrt(mean_square + self.eps)

        # 归一化
        return x.float() * rsqrt

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, model_dim]

        Returns:
            归一化后的张量 [batch_size, seq_len, model_dim]
        """
        # x: [batch_size, seq_len, model_dim]
        normed_x = self._norm(x)

        # 恢复原始数据类型并应用缩放参数
        return normed_x.type_as(x) * self.gamma
