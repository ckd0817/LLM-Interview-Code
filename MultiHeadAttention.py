import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x_query,x_context, mask = None):
        batch_size = x_query.size(0)
        
        # q = k = v = x_query  # Self-Attention
        # q = x_query, k = v = x_context  # Encoder-Decoder cross Attention
        q = self.w_q(x_query)
        
        if x_context is not None:
            k = self.w_k(x_context)
            v = self.w_v(x_context)
        else:
            k = self.w_k(x_query)
            v = self.w_v(x_query)
        
        # 线性变换并分头
        # q, k, v: (B, S, D) -> (B, S, H, D_head) -> (B, H, S, D_head)
        q = q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1,2)
        k = k.view(batch_size, -1, self.num_heads, self.d_head).transpose(1,2)
        v = v.view(batch_size, -1, self.num_heads, self.d_head).transpose(1,2)

        d_k = q.size(-1)
        scores = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # context: (B, H, S, D_head) -> (B, S, H, D_head)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1,2)
        context = context.contiguous()
        # output: (B, S, D)
        output = context.view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output
if __name__ == "__main__":
    # B=2, Seq=10, Dim=64, Heads=8
    x = torch.randn(2, 10, 64)
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    out = mha(x, x, x) # Self-Attention
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}") # 应该还是 (2, 10, 64)