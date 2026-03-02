import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0, dropout=0.0):
        super().__init__()
        
        # Original weight
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.weight.requires_grad = False  # Freeze original weights
        
        # Low-rank adaptation matrices
        # A: (in_features, rank)
        # B: (rank, out_features)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.alpha = alpha
        self.rank = rank
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        
        # initialize LoRA weights
        self.reset_parameters()
        
    def reset_parameters(self):
        
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        
        # Initialize B with zeros
        nn.init.zeros_(self.lora_b.weight)
        
    def forward(self, x):
        with torch.no_grad():
            original_output = self.weight(x)
            
        lora_output = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        
        return original_output + lora_output
    
# --- 测试代码 ---
if __name__ == "__main__":
    x = torch.randn(2, 5, 10) # B, S, D
    # 假设原模型 10 -> 20
    layer = LoRALinear(10, 20, rank=4)
    
    out = layer(x)
    print(f"Output shape: {out.shape}")
    
    # 验证初始状态 LoRA 是否为 0
    # 理论上初始输出应该等于 pretrained 输出
    diff = (out - layer.weight(x)).abs().sum()
    print(f"Diff at init (should be 0): {diff.item()}")