"""
混合专家模型（Mixture of Experts, MoE）

一种稀疏激活的神经网络架构，通过路由机制将输入分配给不同的专家网络。
只有部分专家参与计算，从而在增加模型容量的同时保持计算效率。

参考论文: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    """
    混合专家模型模块

    结构：
    1. Router: 计算每个 token 对每个专家的得分
    2. Top-K 选择: 选择得分最高的 K 个专家
    3. 专家计算: 被选中的专家对 token 进行处理
    4. 加权融合: 根据路由得分加权融合专家输出

    Args:
        d_model: 输入/输出维度
        num_experts: 专家数量
        top_k: 每个 token 激活的专家数量
    """

    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由器：计算每个 token 对每个专家的得分
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # 专家网络列表
        # 每个专家是一个简单的 FFN: Linear -> ReLU -> Linear
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns:
            output: 专家混合输出 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # 展平 batch 和 seq 维度以便处理
        # x_flat: [batch_size * seq_len, d_model]
        x_flat = x.view(-1, d_model)

        # 步骤1: 路由器计算每个 token 对每个专家的得分
        # gate_logits: [batch_size * seq_len, num_experts]
        gate_logits = self.router(x_flat)

        # 步骤2: 选择 top-k 专家
        # weight: 每个 token 对 top-k 专家的权重 [batch_size * seq_len, top_k]
        # indices: 每个 token 选中的 top-k 专家索引 [batch_size * seq_len, top_k]
        weight, indices = torch.topk(gate_logits, self.top_k, dim=-1)

        # 步骤3: 对 top-k 专家的权重进行 softmax 归一化
        # weight: [batch_size * seq_len, top_k]
        weight = F.softmax(weight, dim=-1)

        # 步骤4: 初始化输出张量
        # output: [batch_size * seq_len, d_model]
        output = torch.zeros_like(x_flat)

        # 步骤5: 遍历每个专家，处理选中该专家的 token
        for i, expert in enumerate(self.experts):
            # 找出选择了第 i 个专家的 token
            # mask: bool [batch_size * seq_len, top_k]
            mask = (indices == i)

            # 获取选中第 i 个专家的 token 索引和在 top-k 中的位置
            # token_indices: [num_tokens_using_expert_i]
            # top_k_pos: [num_tokens_using_expert_i]
            token_indices, top_k_pos = torch.where(mask)

            if token_indices.numel() > 0:
                # 提取选中第 i 个专家的 token
                # expert_input: [num_tokens, d_model]
                expert_input = x_flat[token_indices]

                # 计算专家输出
                # expert_output: [num_tokens, d_model]
                expert_output = expert(expert_input)

                # 获取对应的权重
                # expert_weight: [num_tokens]
                expert_weight = weight[token_indices, top_k_pos]

                # 加权专家输出
                # weighted_expert_output: [num_tokens, d_model]
                weighted_expert_output = expert_output * expert_weight.unsqueeze(-1)

                # 将加权专家输出累加到最终输出
                output.index_add_(0, token_indices, weighted_expert_output)

        # 恢复原始形状
        # output: [batch_size, seq_len, d_model]
        return output.view(batch_size, seq_len, d_model)


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入: 16个Token, 维度64; 8个专家, 每个 Token 选 2 个
    x = torch.randn(1, 16, 64)
    moe = MoE(d_model=64, num_experts=8, top_k=2)
    out = moe(x)
    print(f"MoE Output Shape: {out.shape}")  # [1, 16, 64]
