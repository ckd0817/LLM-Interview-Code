import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout_p = 0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_head = d_model % num_heads
        self.num_rep = num_heads % num_kv_heads


        self.w_q = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_head, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_head, bias=False)
        
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout_p)
        
    def repeat_kv(self, x, n_rep):
        """
        x shape: [batch, num_kv_heads, seq_len, d_head]
        output shape: [batch, num_heads, seq_len, d_head]
        """
        batch, num_kv_heads, seq_len, d_heads = x.shape
        if n_rep == 1:
            return x
        
        # 1. 第二维后新增一个维度 [Batch, num_kv_heads, 1 , seq_len, d_head]
        x = x[ : , : , None, : , : ]
        
        # 2. 在新维度上重复n_rep次 [Batch, num_kv_heads, n_rep, seq_len, d_head]
        x = x.expand(batch, num_kv_heads, n_rep, seq_len, d_heads)
        
        # 3. 展平num_kv_heads和n_rep: [Batch, num_kv_heads * n_rep, seq_len, d_head]
        x.reshape(batch,num_kv_heads * n_rep,seq_len,d_heads)
        
        return x
    
    def forward(self, x, mask = None):
        batch_size, seq_len,  __ = x.shape
        
        # 1. 线性变换得到Q, K, V
        q = self.w_q(x) # [batch, seq_len, num_heads * d_head]
        k = self.w_k(x) # [batch, seq_len, num_kv_heads * d_head]
        v = self.w_v(x) # [batch, seq_len, num_kv_heads * d_head]
        
        # 2. 分头
        # q: [batch, seq_len, num_heads, d_head] -> [batch, num_heads, seq_len, d_head]
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1,2)
        # k: [batch, seq_len, num_kv_heads, d_head] -> [batch, num_kv_heads, seq_len, d_head]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.d_head).transpose(1,2)
        # v: [batch, seq_len, num_kv_heads, d_head] -> [batch, num_kv_heads, seq_len, d_head]
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.d_head).transpose(1,2)

        
        # 3. 重复K,V以匹配Q的头数
        k = self.repeat_kv(k, self.num_rep) # [batch, num_heads, seq_len, d_head]
        v = self.repeat_kv(v, self.num_rep) # [batch, num_heads , seq_len, d_head]
        
        
        # 4. 计算注意力得分
        # scores: [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_head) 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. 计算上下文向量
        # attn_weights: [batch, num_heads, seq_len, seq_len]
        # v: [batch, num_heads, seq_len, d_head]
        # context: [batch, num_heads, seq_len, d_head]
        context = torch.matmul(attn_weights, v) 
        
        context = context.transpose(1,2) # [batch, seq_len, num_heads, d_head]
        context = context.contiguous()
        
        # 6. 拼接多头
        # output: [batch, seq_len, d_model]
        output = context.view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        return output


        
        