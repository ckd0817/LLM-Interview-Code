"""
层归一化（Layer Normalization）

Transformer 中常用的归一化技术，在每个样本的特征维度上进行归一化。
与 BatchNorm 不同，LayerNorm 不依赖 batch 统计量，适用于可变长度序列。

参考论文: Layer Normalization
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    层归一化模块

    公式: LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

    Args:
        model_dim: 归一化的特征维度
        eps: 数值稳定性常数，防止除零，默认 1e-5
    """

    def __init__(self, model_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_dim))   # 可学习的缩放参数
        self.beta = nn.Parameter(torch.zeros(model_dim))   # 可学习的偏移参数

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, model_dim]

        Returns:
            归一化后的张量 [batch_size, seq_len, model_dim]
        """
        # x: [batch_size, seq_len, model_dim]

        # 计算均值
        # mean: [batch_size, seq_len, 1]
        mean = x.mean(-1, keepdim=True)

        # 计算方差
        # 【面试大坑】torch.var 默认是 unbiased=True（除以 N-1）
        # 但 LayerNorm 的定义通常是除以 N（unbiased=False）
        # var: [batch_size, seq_len, 1]
        var = x.var(-1, keepdim=True, unbiased=False)

        # 归一化：零均值、单位方差
        # x_normalized: [batch_size, seq_len, model_dim]
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # 应用可学习的仿射变换
        return x_normalized * self.gamma + self.beta
