import torch

def compute_grpo_advantages(rewards):
    """
    计算 GRPO 中的优势函数, 去掉了 Critic, 用组内归一化代替 Advantage。 
    
    参数:
        rewards (torch.Tensor): [Batch, Group_Size]
        
    返回:
        advantages (torch.Tensor): [Batch, Group_Size]
    """
    mean = rewards.mean(dim=-1,keepdim=True)
    std = rewards.std(dim=-1,keepdim=True)
    
    advantages = (rewards - mean) / (std + 1e-8)
    
    return advantages

def grpo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2, beta=0.01,ref_kl=None):
    """
    计算 GRPO 损失函数。

    参数:
    - old_log_probs: 旧策略的对数概率张量。
    - new_log_probs: 新策略的对数概率张量。
    - advantages: 优势估计张量。
    - clip_epsilon: GRPO 的截断参数。
    - beta: KL 散度惩罚系数。
    - ref_kl: 参考 KL 散度, 如果提供则使用该值计算 KL 惩罚。

    返回:
    - loss: 计算得到的 GRPO 损失。
    """
    
    # 1. 计算重要性采样比率
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 2. 计算截断比率
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    
    # 3. 计算代理损失
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    
    policy_loss = -torch.min(surrogate1, surrogate2)
    
    # 4. GRPO 特有的 KL 正则项 (DeepSeek 做法)
    # DPO 把 KL 藏在 Loss 里，GRPO 通常显式地加一个 KL 惩罚
    # loss = policy_loss + beta * kl(policy || ref)
    if ref_kl is not None:
        return (policy_loss + beta * ref_kl).mean()
    
    return policy_loss.mean()

def compute_kl_penalty(log_probs, ref_log_probs):
    """
    计算 KL 散度惩罚项。

    参数:
    - log_probs: 当前策略的对数概率张量。
    - ref_log_probs: 参考策略的对数概率张量。
    """
    # Schulman估计器
    # KL(P || Q) = E_P [ log P(x) - log Q(x) ] 
    # = E_P[exp(log Q(x) - log P(x)) - (log Q(x) - log P(x)) - 1]
    # = E_P[Q(x)/P(x)] - E_P[log Q(x) - log P(x)] - E_P[1]
    # = 1 - E_P[log Q(x) - log P(x)] - 1
    # = - E_P[log Q(x) - log P(x)]
    # = E_P[log P(x) - log Q(x)]
    ratio = torch.exp(ref_log_probs - log_probs)
    kl = ratio - (ref_log_probs - log_probs) - 1
    
    return kl.mean()