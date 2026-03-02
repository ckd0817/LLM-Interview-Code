"""
预训练损失（Pretrain Loss）

用于大语言模型预训练的标准因果语言模型损失。
采用下一个词预测（Next Token Prediction）任务。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PretrainLoss(nn.Module):
    """
    预训练损失模块

    计算因果语言模型的交叉熵损失。
    通过预测下一个词来训练语言模型。

    Args:
        ignore_index: 忽略的标签索引，不计入损失计算，默认 -100
    """

    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        前向传播

        Args:
            logits: 模型输出的未归一化对数概率 [batch_size, seq_len, vocab_size]
            labels: 真实词元索引 [batch_size, seq_len]

        Returns:
            loss: 标量损失值
        """
        # 步骤1: 移位操作（Shift）
        # 因果语言模型：用前面的词预测下一个词
        # logits 去掉最后一个位置：该位置没有下一个词可预测
        # [batch_size, seq_len, vocab_size] -> [batch_size, seq_len-1, vocab_size]
        shifted_logits = logits[:, :-1, :].contiguous()

        # labels 去掉第一个位置：第一个位置没有前文
        # [batch_size, seq_len] -> [batch_size, seq_len-1]
        shifted_labels = labels[:, 1:].contiguous()

        # 步骤2: 展平张量
        # CrossEntropyLoss 期望输入为二维张量 [num_samples, num_classes] 和一维张量 [num_samples]
        batch_size, seq_length, vocab_size = shifted_logits.size()

        # flattened_logits: [batch_size * (seq_len-1), vocab_size]
        flattened_logits = shifted_logits.view(-1, vocab_size)

        # flattened_labels: [batch_size * (seq_len-1)]
        flattened_labels = shifted_labels.view(-1)

        # 步骤3: 计算交叉熵损失
        # ignore_index=-100 的位置不参与损失计算
        loss = F.cross_entropy(flattened_logits, flattened_labels, ignore_index=self.ignore_index)

        return loss
