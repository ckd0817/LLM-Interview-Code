import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, labels, prompt_lengths):
        """
        logits: Tensor of shape (batch_size, seq_length, vocab_size)
        labels: Tensor of shape (batch_size, seq_length)
        prompt_lengths: Tensor of shape (batch_size,) indicating the length of the prompt in each sequence
        """
        
        # 1. 构造 masked labels
        masked_labels = labels.clone()
        for batch_idx, prompt_length in enumerate(prompt_lengths):
            masked_labels[batch_idx, :prompt_length] = -100  # 设置为 ignore_index
            
        # 2. Shift
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = masked_labels[:, 1:].contiguous()
        
        # 3. Flatten
        batch_size, seq_length, vocab_size = shifted_logits.size()
        flattened_logits = shifted_logits.view(-1, vocab_size)  # [B*(S-1), V]
        flattened_labels = shifted_labels.view(-1)              # [B*(S-1)]
        # 4. 计算交叉熵损失
        loss = F.cross_entropy(flattened_logits, flattened_labels, ignore_index=-100)
        return loss