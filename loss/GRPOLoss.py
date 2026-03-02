"""
组相对策略优化损失（Group Relative Policy Optimization Loss）

GRPO 是 DeepSeek 提出的一种 RLHF 算法，是 PPO 的简化变体。
主要特点是去掉了 Critic 网络，使用组内归一化来计算优势函数。

参考论文: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
"""

import torch


def compute_grpo_advantages(rewards):
    """
    计算 GRPO 中的优势函数

    GRPO 去掉了 Critic 网络，使用组内归一化代替传统的优势估计。
    对于每个问题生成一组回答，然后计算组内的相对优势。

    公式: A = (R - mean(R)) / (std(R) + eps)

    Args:
        rewards: 每个回答的奖励值 [batch_size, group_size]

    Returns:
        advantages: 归一化后的优势值 [batch_size, group_size]
    """
    # 计算组内均值
    # mean: [batch_size, 1]
    mean = rewards.mean(dim=-1, keepdim=True)

    # 计算组内标准差
    # std: [batch_size, 1]
    std = rewards.std(dim=-1, keepdim=True)

    # 归一化得到优势值
    # advantages: [batch_size, group_size]
    advantages = (rewards - mean) / (std + 1e-8)

    return advantages


def grpo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2, beta=0.01, ref_kl=None):
    """
    计算 GRPO 损失

    GRPO 的损失函数与 PPO 类似，但通常包含显式的 KL 散度惩罚项。
    公式: L = E[min(r_t * A_t, clip(r_t) * A_t)] + β * KL(π || π_ref)

    Args:
        old_log_probs: 旧策略的对数概率 [batch_size]
        new_log_probs: 新策略的对数概率 [batch_size]
        advantages: 优势估计值 [batch_size]
        clip_epsilon: 截断参数 ε，默认 0.2
        beta: KL 散度惩罚系数，默认 0.01
        ref_kl: 当前策略与参考策略的 KL 散度（可选）[batch_size]

    Returns:
        loss: GRPO 损失值（标量）
    """
    # 步骤1: 计算重要性采样比率
    # r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 步骤2: 计算截断后的比率
    # 将比率限制在 [1 - ε, 1 + ε] 范围内
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    # 步骤3: 计算代理损失
    # surrogate1: 未截断的目标 r_t * A_t
    surrogate1 = ratio * advantages

    # surrogate2: 截断后的目标 clip(r_t) * A_t
    surrogate2 = clipped_ratio * advantages

    # 取两者中的较小值
    policy_loss = -torch.min(surrogate1, surrogate2)

    # 步骤4: GRPO 特有的 KL 正则项（DeepSeek 做法）
    # DPO 把 KL 藏在 Loss 里，GRPO 通常显式地加一个 KL 惩罚
    # loss = policy_loss + β * KL(π || π_ref)
    if ref_kl is not None:
        return (policy_loss + beta * ref_kl).mean()

    return policy_loss.mean()


def compute_kl_penalty(log_probs, ref_log_probs):
    """
    计算 KL 散度惩罚项

    使用 Schulman 估计器计算 KL 散度，该方法具有较低的方差。

    公式: KL(P || Q) ≈ E_P[exp(log Q - log P) - (log Q - log P) - 1]

    推导:
    KL(P || Q) = E_P[log P(x) - log Q(x)]
               = E_P[exp(log Q(x) - log P(x)) - (log Q(x) - log P(x)) - 1]
    （通过 Taylor 展开得到无偏估计）

    Args:
        log_probs: 当前策略的对数概率 [batch_size]
        ref_log_probs: 参考策略的对数概率 [batch_size]

    Returns:
        kl: KL 散度的估计值（标量）
    """
    # Schulman 估计器
    # ratio = exp(log Q - log P) = Q / P
    ratio = torch.exp(ref_log_probs - log_probs)

    # KL = E_P[ratio - log_ratio - 1]
    # 其中 log_ratio = log Q - log P
    kl = ratio - (ref_log_probs - log_probs) - 1

    return kl.mean()
