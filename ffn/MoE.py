import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self,d_model, num_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])
    def forward(self, x):
        """
        x shape: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        x_flat = x.view(-1, d_model) # [batch * seq_len, d_model]
        
        # 1. 路由器计算每个token对每个专家的打分
        gate_logits = self.router(x_flat) # [batch * seq_len, num_experts]
        
        # 2. 选取top-k专家
        # weight: 每个token对top-k专家的权重 [batch * seq_len, top_k]
        # indices: 每个token选中的top-k专家的索引 [batch * seq_len, top_k]
        weight, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # 3. 对top-k专家的权重进行softmax归一化
        weight = F.softmax(weight, dim=-1) # [batch * seq_len, top_k]
        
        # 4. 初始化输出张量
        output = torch.zeros_like(x_flat) # [batch * seq_len, d_model]
        
        # 5. 对每个token的top-k专家进行计算并加权求和
        for i, expert in enumerate(self.experts):
            # 找出选择了第i个专家的token
            mask = (indices == i) # bool: [batch * seq_len, top_k]
            token_indices, top_k_pos = torch.where(mask) # token_indices: [num_tokens_using_expert_i], top_k_pos: [expert_position_in_top_k]
            
            if token_indices.numel() > 0:
                # 提取选中第i个专家的token输入
                expert_input = x_flat[token_indices] # [num_tokens_using_expert_i, d_model]
                # 计算专家输出
                expert_output = expert(expert_input) # [num_tokens_using_expert_i, d_model]
                # 获取对应的权重
                expert_weight = weight[token_indices, top_k_pos] # [num_tokens_using_expert_i]
                # 加权求和到输出
                weighted_expert_output = expert_output * expert_weight.unsqueeze(-1) # [num_tokens_using_expert_i, d_model]
                
                # 将加权专家输出累加到最终输出
                output.index_add_(0, token_indices, weighted_expert_output)
        
        return output.view(batch_size, seq_len, d_model)
    
# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入: 16个Token, 维度64; 8个专家, 每个Token选2个
    x = torch.randn(1, 16, 64)
    moe = MoE(d_model=64, num_experts=8, top_k=2)
    out = moe(x)
    print(f"MoE Output Shape: {out.shape}") # [1, 16, 64]