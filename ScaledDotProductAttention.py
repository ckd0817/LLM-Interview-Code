import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self,q,k,v,mask = None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights,v)

        return output,attn_weights

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入: Batch=2, Heads=4, Seq_Len=8, Dim=64
    B, H, S, D = 2, 4, 8, 64
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    # 构造一个 Causal Mask (下三角矩阵)，模拟 GPT 生成过程
    # shape: (S, S), 上三角为 0, 下三角为 1
    causal_mask = torch.tril(torch.ones(S, S)).view(1, 1, S, S)
    
    attention_layer = ScaledDotProductAttention()
    output, weights = attention_layer(q, k, v, mask=causal_mask)
    
    print(f"Output shape: {output.shape}") # Should be (2, 4, 8, 64)
    print(f"Weights shape: {weights.shape}") # Should be (2, 4, 8, 8)
    
    # 验证 Mask 是否生效：查看第一个样本第一个头的第一行
    # 理论上只有第1个位置有值，后面全是0
    print("\nCheck Causal Masking (Row 0 should only attend to Col 0):")
    print(weights[0, 0, 0, :])