import torch
import torch.nn as nn
import torch.nn.functional as F

class PretrainLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        
    def forward(self, logits, labels):
        """
        logits: Tensor of shape (batch_size, seq_length, vocab_size)
        labels: Tensor of shape (batch_size, seq_length)
        """
        
        # 1. Shift 
        # 预测值logits 去掉最后一个：最后一个位置没有下一个词
        # [B, S, V] -> [B, S-1, V]
        shifted_logits = logits[:, :-1, :].contiguous()
        # 真实值labels 去掉第一个：第一个位置没有前一个词
        # [B, S] -> [B, S-1]
        shifted_labels = labels[:, 1:].contiguous()
        # 2. Flatten
        # CrossEntropyLoss 期望输入为二维张量 [N, C] 和一维张量 [N]
        batch_size, seq_length, vocab_size = shifted_logits.size()
        flattened_logits = shifted_logits.view(-1, vocab_size)  # [B*(S-1), V]
        flattened_labels = shifted_labels.view(-1)              # [B*(S-1)]
        
        loss = F.cross_entropy(flattened_logits, flattened_labels, ignore_index=self.ignore_index)
        
        return loss