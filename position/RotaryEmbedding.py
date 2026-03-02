import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048,theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        cos, sin = self.precompute_freqs(dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        
    def precompute_freqs(self, dim, max_seq_len, theta):
        # inv_freqs shape: (dim/2)
        inv_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        # t shape: (max_seq_len)
        t = torch.arange(max_seq_len, device = inv_freqs.device, dtype=torch.float32)
        
        angles = torch.outer(t, inv_freqs) # angles shape: (max_seq_len, dim/2)
        
        angles = torch.cat((angles, angles), dim=-1)  # angles shape: (max_seq_len, dim)
        return angles.cos(), angles.sin()
    
    def forward(self, xq, xk):
        # xq, xk shape: (batch_size, seq_len, num_heads, head_dim)
        seq_len = xq.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, self.dim) # (1, seq_len, 1, dim)
        sin = self.sin[:seq_len].view(1, seq_len, 1, self.dim) # (1, seq_len, 1, dim)
        
        def rotate_half(x):
            x1, x2 = torch.chunk(x, 2, dim=-1) # x1, x2 shape: (..., dim/2)
            return torch.cat((-x2, x1), dim=-1) # (..., dim)
        
        # xq * cos : [x1,x2] * cos, rotate_half(xq) * sin : [-x2,x1] * sin
        # Final: [x1*cos - x2*sin, x2*cos + x1*sin]
        # [x1';x2'] = [cos, sin; -sin, cos] [x1; x2]
        
        xq_rotated = (xq * cos) + (rotate_half(xq) * sin)
        xk_rotated = (xk * cos) + (rotate_half(xk) * sin)
        
        return xq_rotated, xk_rotated
    
    