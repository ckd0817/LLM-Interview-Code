import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, d_latent, d_rope, dropout_p):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_latent = d_latent
        self.d_rope = d_rope
        
        # 1.kv projection
        self.kv_down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.kv_up_proj = nn.Linear(d_latent,num_heads * (d_head + d_rope + d_head), bias=False)
        # 2.q projection
        self.q_down_proj = nn.Linear(d_model,d_latent,bias=False)
        self.q_up_proj = nn.Linear(d_latent,num_heads * (d_head + d_rope), bias=False)
        # 3. output projection
        self.o_proj = nn.Linear(num_heads * d_head, d_model, bias=False)
        
    def forward(self, x , mask= None):
        batch_size, seq_len, _ = x.size()
        
        # kv projection
        kv_latent = self.kv_down_proj(x)  # (batch_size, seq_len,  d_latent) for inference efficiency
        kv_full = self.kv_up_proj(kv_latent)  # (batch_size, seq_len, num_heads * (d_head + d_rope + d_head))
        kv_full = kv_full.view(batch_size,seq_len,self.num_heads, - 1)
        # split content and rope
        # k_content: (batch_size, seq_len, num_heads, d_head)
        # k_rope: (batch_size, seq_len, num_heads, d_rope)
        # v_content: (batch_size, seq_len, num_heads, d_head)
        k_content, k_rope, v_content = torch.split(kv_full, [self.d_head, self.d_rope, self.d_head], dim=-1)  
        
        
        # q projection
        q_latent = self.q_down_proj(x)  # (batch_size, seq_len, d_latent)
        q_full = self.q_up_proj(q_latent)  # (batch_size, seq_len, num_heads * (d_head + d_rope))
        q_full = q_full.view(batch_size,seq_len,self.num_heads,self.d_head + self.d_rope)
        # split content and rope    
        q_content, q_rope = torch.split(q_full, [self.d_head, self.d_rope], dim=-1)
        
        # Apply RoPE to q_rope and k_rope
        q_rope, k_rope = self.apply_rope(q_rope, k_rope)
        
        # Combine content and rope parts
        q = torch.cat([q_content, q_rope], dim=-1)  # (batch_size, seq_len, num_heads, d_head + d_rope)
        q.transpose_(1, 2)  # (batch_size, num_heads, seq_len, d_head + d_rope)
        k = torch.cat([k_content, k_rope], dim=-1)  # (batch_size, seq_len, num_heads, d_head + d_rope)
        k.transpose_(1, 2)  # (batch_size, num_heads, seq_len, d_head + d_rope)
        v = v_content.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_head)
        
        # Scaled dot-product attention
        # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head + self.d_rope)  
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        context = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, d_head)
        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_head)  # (batch_size, seq_len, num_heads * d_head)
        output = self.o_proj(output)  # (batch_size, seq_len, d_model)
        return output