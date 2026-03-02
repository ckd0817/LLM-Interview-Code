"""
近端策略优化损失（Proximal Policy Optimization Loss）

PPO 是一种基于置信域的策略梯度算法，通过截断重要性采样比率
来限制策略更新的幅度，保证训练的稳定性。

参考论文: Proximal Policy Optimization Algorithms
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2):
    """
    计算 PPO 截断损失

    PPO 通过截断重要性采样比率来限制策略更新的幅度：
    L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

    其中 r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) 是重要性采样比率。

    Args:
        old_log_probs: 旧策略的对数概率 [batch_size]
        new_log_probs: 新策略的对数概率 [batch_size]
        advantages: 优势估计值 [batch_size]
        clip_epsilon: 截断参数 ε，默认 0.2

    Returns:
        loss: PPO 截断损失值（标量）
    """
    # 步骤1: 计算重要性采样比率
    # r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) = exp(log π_θ - log π_θ_old)
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 步骤2: 计算截断后的比率
    # 将比率限制在 [1 - ε, 1 + ε] 范围内
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    # 步骤3: 计算代理损失
    # surrogate1: 未截断的目标 r_t * A_t
    surrogate1 = ratio * advantages

    # surrogate2: 截断后的目标 clip(r_t) * A_t
    surrogate2 = clipped_ratio * advantages

    # 步骤4: 取两者中的较小值（保守更新）
    # 取最小值使得：当 A > 0 时，限制奖励增长；当 A < 0 时，限制惩罚增长
    loss = -torch.mean(torch.min(surrogate1, surrogate2))

    return loss


def plot_ppo_clip():
    """
    绘制 PPO 截断函数的可视化图表

    展示 PPO 在不同优势值下如何限制策略更新的幅度。
    """
    # 设定 r 的范围 (0 到 2)
    r = np.linspace(0, 2, 200)
    epsilon = 0.2

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 情况 1: Advantage > 0 (这是一个好动作) ---
    A_pos = 1.0
    # 1. 未截断的收益 (r * A)
    obj_unclipped_pos = r * A_pos
    # 2. 截断的收益 (clip(r) * A)
    obj_clipped_pos = np.clip(r, 1 - epsilon, 1 + epsilon) * A_pos
    # 3. PPO 最终收益 (取最小值 min)
    obj_ppo_pos = np.minimum(obj_unclipped_pos, obj_clipped_pos)

    # 绘图
    ax1.plot(r, obj_unclipped_pos, 'g--', label='Unclipped (r*A)', alpha=0.5)
    ax1.plot(r, obj_clipped_pos, 'b--', label='Clipped (clip*A)', alpha=0.5)
    ax1.plot(r, obj_ppo_pos, 'r-', linewidth=3, label='PPO Reward (Min)')

    # 标注区域
    ax1.set_title(f'Case 1: Advantage > 0 (Good Action)\nLimit Reward for large change')
    ax1.axvline(x=1 + epsilon, color='k', linestyle=':', label='1+epsilon')
    ax1.set_xlabel('Probability Ratio r_t')
    ax1.set_ylabel('Reward L')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(1.3, 1.05, 'Gradient = 0\n(Stop Updating)', color='red', fontweight='bold')

    # --- 情况 2: Advantage < 0 (这是一个坏动作) ---
    A_neg = -1.0
    # 1. 未截断的收益
    obj_unclipped_neg = r * A_neg
    # 2. 截断的收益
    obj_clipped_neg = np.clip(r, 1 - epsilon, 1 + epsilon) * A_neg
    # 3. PPO 最终收益 (取最小值 min)
    # 注意：因为 A 是负的，min 会发挥"悲观"作用
    obj_ppo_neg = np.minimum(obj_unclipped_neg, obj_clipped_neg)

    # 绘图
    ax2.plot(r, obj_unclipped_neg, 'g--', label='Unclipped (r*A)', alpha=0.5)
    ax2.plot(r, obj_clipped_neg, 'b--', label='Clipped (clip*A)', alpha=0.5)
    ax2.plot(r, obj_ppo_neg, 'r-', linewidth=3, label='PPO Reward (Min)')

    # 标注区域
    ax2.set_title(f'Case 2: Advantage < 0 (Bad Action)\nLimit Penalty for large change')
    ax2.axvline(x=1 - epsilon, color='k', linestyle=':', label='1-epsilon')
    ax2.set_xlabel('Probability Ratio r_t')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.1, -0.7, 'Gradient = 0\n(Stop Updating)', color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()


# 运行绘图
plot_ppo_clip()
