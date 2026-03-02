import torch

def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2):
    """
    Compute the PPO clipped surrogate loss.

    Parameters:
    - old_log_probs: Tensor of log probabilities from the old policy.
    - new_log_probs: Tensor of log probabilities from the new policy.
    - advantages: Tensor of advantage estimates.
    - clip_epsilon: Clipping parameter for PPO.

    Returns:
    - loss: The computed PPO clipped surrogate loss.
    """
    

    # 1. Compute Importance Sampling Ratios
    # r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 2. Compute Clipped Ratios
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    
    # 3. Compute Surrogate Losses
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    loss = -torch.mean(torch.min(surrogate1, surrogate2))
    return loss

import numpy as np
import matplotlib.pyplot as plt

def plot_ppo_clip():
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
    ax1.axvline(x=1+epsilon, color='k', linestyle=':', label='1+epsilon')
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
    # 注意：因为 A 是负的，min 会发挥“悲观”作用
    obj_ppo_neg = np.minimum(obj_unclipped_neg, obj_clipped_neg)

    # 绘图
    ax2.plot(r, obj_unclipped_neg, 'g--', label='Unclipped (r*A)', alpha=0.5)
    ax2.plot(r, obj_clipped_neg, 'b--', label='Clipped (clip*A)', alpha=0.5)
    ax2.plot(r, obj_ppo_neg, 'r-', linewidth=3, label='PPO Reward (Min)')

    # 标注区域
    ax2.set_title(f'Case 2: Advantage < 0 (Bad Action)\nLimit Penalty for large change')
    ax2.axvline(x=1-epsilon, color='k', linestyle=':', label='1-epsilon')
    ax2.set_xlabel('Probability Ratio r_t')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.1, -0.7, 'Gradient = 0\n(Stop Updating)', color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()

# 运行绘图
plot_ppo_clip()