"""
LoRA 线性层（Low-Rank Adaptation Linear）

一种高效的参数高效微调（PEFT）方法，通过低秩分解来近似权重更新。
只训练少量参数即可达到接近全量微调的效果。

参考论文: LoRA: Low-Rank Adaptation of Large Language Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    LoRA 线性层模块

    将权重更新分解为两个低秩矩阵的乘积: ΔW = B @ A
    前向传播: y = W @ x + (B @ A) @ x * scaling

    优势:
    - 大幅减少可训练参数（rank << min(in_features, out_features)）
    - 原始权重冻结，支持即插即用
    - 训练完成后可将 LoRA 权重合并到原始权重中

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: LoRA 的秩（低秩矩阵的维度），默认 8
        alpha: LoRA 的缩放因子，默认 1.0
        dropout: Dropout 概率，默认 0.0
    """

    def __init__(self, in_features, out_features, rank=8, alpha=1.0, dropout=0.0):
        super().__init__()

        # 原始预训练权重（冻结）
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.weight.requires_grad = False  # 冻结原始权重

        # LoRA 低秩适应矩阵
        # A: [in_features, rank] - 下投影
        # B: [rank, out_features] - 上投影
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        self.alpha = alpha
        self.rank = rank

        # 缩放因子：alpha / rank
        # 用于平衡 LoRA 输出和原始输出的比例
        self.scaling = self.alpha / rank

        self.dropout = nn.Dropout(dropout)

        # 初始化 LoRA 权重
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化 LoRA 权重

        - A 使用 Kaiming 均匀初始化
        - B 初始化为零，确保初始状态时 LoRA 输出为零
        """
        # A 使用 Kaiming 均匀初始化
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))

        # B 初始化为零，确保初始输出等于预训练模型输出
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, in_features]

        Returns:
            输出张量 [batch_size, seq_len, out_features]
        """
        # 步骤1: 计算原始输出（不计算梯度）
        # original_output: [batch_size, seq_len, out_features]
        with torch.no_grad():
            original_output = self.weight(x)

        # 步骤2: 计算 LoRA 增量输出
        # x -> dropout -> A -> B -> scaling
        # lora_output: [batch_size, seq_len, out_features]
        lora_output = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling

        # 步骤3: 合并原始输出和 LoRA 增量
        return original_output + lora_output


# --- 测试代码 ---
if __name__ == "__main__":
    x = torch.randn(2, 5, 10)  # batch_size, seq_len, in_features
    # 假设原模型 10 -> 20
    layer = LoRALinear(10, 20, rank=4)

    out = layer(x)
    print(f"Output shape: {out.shape}")

    # 验证初始状态 LoRA 是否为 0
    # 理论上初始输出应该等于 pretrained 输出
    diff = (out - layer.weight(x)).abs().sum()
    print(f"Diff at init (should be 0): {diff.item()}")
