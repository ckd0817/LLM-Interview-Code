"""
SwiGLU 前馈神经网络

一种结合了 Swish 激活函数和 GLU（Gated Linear Unit）门控机制的前馈网络。
相比传统 FFN，SwiGLU 在保持相似参数量的情况下具有更好的性能。

参考论文: GLU Variants Improve Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈神经网络模块

    结构: output = Down(Gate(x) * Up(x))
    其中 Gate 使用 SiLU (Swish) 激活函数，实现门控机制。

    相比传统 FFN 的 ReLU 激活，SwiGLU 的门控机制可以更好地捕捉复杂的非线性关系。

    Args:
        d_model: 输入/输出维度
        intermediate_dim: 中间层维度（通常是 d_model 的 8/3 倍）
    """

    def __init__(self, d_model, intermediate_dim):
        super().__init__()
        self.d_model = d_model
        self.intermediate_dim = intermediate_dim

        # 门控分支（使用 SiLU 激活）
        self.w_gate = nn.Linear(d_model, intermediate_dim, bias=False)

        # 上投影分支（无激活）
        self.w_up = nn.Linear(d_model, intermediate_dim, bias=False)

        # 下投影（恢复到原始维度）
        self.w_down = nn.Linear(intermediate_dim, d_model, bias=False)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # SwiGLU 公式: Down(SiLU(Gate(x)) * Up(x))
        #
        # 步骤1: 计算门控值（使用 SiLU 激活）
        # gate: [batch_size, seq_len, d_model] -> [batch_size, seq_len, intermediate_dim]
        gate = F.silu(self.w_gate(x))

        # 步骤2: 计算上投影值（无激活）
        # up: [batch_size, seq_len, d_model] -> [batch_size, seq_len, intermediate_dim]
        up = self.w_up(x)

        # 步骤3: 门控相乘（GLU 机制）
        # activated_feature: [batch_size, seq_len, intermediate_dim]
        activated_feature = gate * up

        # 步骤4: 下投影恢复维度
        # output: [batch_size, seq_len, intermediate_dim] -> [batch_size, seq_len, d_model]
        output = self.w_down(activated_feature)

        return output
