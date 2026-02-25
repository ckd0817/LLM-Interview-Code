import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        
    def _norm(self,x):
        # x: [batch_size, seq_len, dim]
        # mean_square: [batch_size, seq_len, 1]
        mean_square = x.float().pow(2).mean(-1,keepdim=True)
        # rsqrt = 1/sqrt(x)
        rsqrt = torch.rsqrt(mean_square + self.eps)
        
        return x.float() * rsqrt
    
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        normed_x = self._norm(x)
        return normed_x.type_as(x) * self.gamma