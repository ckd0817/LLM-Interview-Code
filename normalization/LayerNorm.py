import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self,x):
        # x: (batch_size, seq_len, d_model)
        mean = x.mean(-1, keepdim=True)
        # 面试大坑：torch.var 默认是 unbiased=True (除以 N-1)
        # 但 LayerNorm 的定义通常是除以 N (unbiased=False)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return x_normalized * self.gamma + self.beta