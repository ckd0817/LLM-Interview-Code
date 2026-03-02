"""
监督微调损失（Supervised Fine-Tuning Loss）

用于大语言模型监督微调的损失函数。
与预训练损失类似，但支持屏蔽 prompt 部分，只计算 response 的损失。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFTLoss(nn.Module):
    """
    监督微调损失模块

    在 SFT 阶段，我们通常只希望计算 response 部分的损失，
    而不计算 prompt 部分的损失。该模块支持通过 prompt_lengths 来屏蔽 prompt。

    Args:
        无
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, prompt_lengths):
        """
        前向传播

        Args:
            logits: 模型输出的未归一化对数概率 [batch_size, seq_len, vocab_size]
            labels: 真实词元索引 [batch_size, seq_len]
            prompt_lengths: 每个样本的 prompt 长度 [batch_size]

        Returns:
            loss: 标量损失值
        """
        # 步骤1: 构造 masked labels
        # 将 prompt 部分的标签设为 -100（ignore_index）
        masked_labels = labels.clone()
        for batch_idx, prompt_length in enumerate(prompt_lengths):
            masked_labels[batch_idx, :prompt_length] = -100  # 设置为 ignore_index

        # 步骤2: 移位操作（Shift）
        # 预测下一个词：logits 去掉最后一个，labels 去掉第一个
        # shifted_logits: [batch_size, seq_len-1, vocab_size]
        shifted_logits = logits[:, :-1, :].contiguous()

        # shifted_labels: [batch_size, seq_len-1]
        shifted_labels = masked_labels[:, 1:].contiguous()

        # 步骤3: 展平张量
        batch_size, seq_length, vocab_size = shifted_logits.size()

        # flattened_logits: [batch_size * (seq_len-1), vocab_size]
        flattened_logits = shifted_logits.view(-1, vocab_size)

        # flattened_labels: [batch_size * (seq_len-1)]
        flattened_labels = shifted_labels.view(-1)

        # 步骤4: 计算交叉熵损失
        # ignore_index=-100 的位置（prompt 部分）不参与损失计算
        loss = F.cross_entropy(flattened_logits, flattened_labels, ignore_index=-100)

        return loss
