import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, d_model, intermediate_dim):
        super().__init__()
        self.d_model = d_model
        
        # intermediate_dim通常是d_model的4倍
        self.intermediate_dim = intermediate_dim
        
        self.w_up = nn.Linear(d_model, intermediate_dim) # Up
        self.w_down = nn.Linear(intermediate_dim, d_model) # Down
        
    def forward(self, x):
        # FFN : ReLU(W_up(x)) * W_down(x)
        up = F.relu(self.w_up(x))
        output = self.w_down(up)
        return output