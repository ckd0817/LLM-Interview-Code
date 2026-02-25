import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, intermediate_dim):
        super().__init__()
        self.d_model = d_model
        self.intermediate_dim = intermediate_dim
        
        self.w1 = nn.Linear(d_model, intermediate_dim, bias=False) # Gate
        self.w2 = nn.Linear(d_model, intermediate_dim, bias=False) # Up
        self.w3 = nn.Linear(intermediate_dim, d_model, bias=False) # Down
        
    def forward(self, x):
        # SwiGLU : (SiLU((W_gate(x))) * W_up(x)) * W_down(x)
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        activated_feature = gate * up
        output = self.w3(activated_feature)
        return output
        