"""
标准前馈神经网络（Feed-Forward Network）

Transformer 中的标准 FFN 模块，由两个线性变换和一个 ReLU 激活函数组成。
公式: FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """
    标准前馈神经网络模块

    由两个线性层组成，中间使用 ReLU 激活函数。
    intermediate_dim 通常是 model_dim 的 4 倍。

    Args:
        model_dim: 输入/输出维度
        intermediate_dim: 中间层维度（通常是 model_dim 的 4 倍）
    """

    def __init__(self, model_dim, intermediate_dim):
        super().__init__()
        self.model_dim = model_dim

        # intermediate_dim 通常是 model_dim 的 4 倍
        self.intermediate_dim = intermediate_dim

        # 上投影：model_dim -> intermediate_dim
        self.w_up = nn.Linear(model_dim, intermediate_dim)

        # 下投影：intermediate_dim -> model_dim
        self.w_down = nn.Linear(intermediate_dim, model_dim)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, model_dim]

        Returns:
            输出张量 [batch_size, seq_len, model_dim]
        """
        # FFN 公式: ReLU(W_up(x)) * W_down(x)
        #
        # 步骤1: 上投影并应用 ReLU 激活
        # up: [batch_size, seq_len, model_dim] -> [batch_size, seq_len, intermediate_dim]
        up = F.relu(self.w_up(x))

        # 步骤2: 下投影恢复维度
        # output: [batch_size, seq_len, intermediate_dim] -> [batch_size, seq_len, model_dim]
        output = self.w_down(up)

        return output
