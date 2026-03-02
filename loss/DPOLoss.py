"""
直接偏好优化损失（Direct Preference Optimization Loss）

DPO 是一种不使用奖励模型的 RLHF 替代方案。
通过直接优化人类偏好数据来对齐语言模型，避免了训练奖励模型的复杂性。

参考论文: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
"""

import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1, label_smoothing=0.0):
    """
    计算 DPO 损失

    DPO 的核心思想是：将奖励函数参数化为策略和参考策略的对数比率，
    然后直接在偏好数据上优化这个隐式奖励。

    公式: L_DPO = -E[log sigmoid(β * (log(π_θ(y_w|x) / π_ref(y_w|x)) - log(π_θ(y_l|x) / π_ref(y_l|x))))]

    Args:
        policy_chosen_logps: 策略模型对 chosen 回复的对数概率 [batch_size]
        policy_rejected_logps: 策略模型对 rejected 回复的对数概率 [batch_size]
        ref_chosen_logps: 参考模型对 chosen 回复的对数概率 [batch_size]
        ref_rejected_logps: 参考模型对 rejected 回复的对数概率 [batch_size]
        beta: KL 散度的缩放因子，控制对参考模型的偏离程度，默认 0.1
        label_smoothing: 标签平滑系数，默认 0.0

    Returns:
        loss: DPO 损失值（标量）
    """
    # 步骤1: 计算对数比率（隐式奖励）
    # chosen_ratio = log(π_θ(y_w|x) / π_ref(y_w|x))
    chosen_ratio = policy_chosen_logps - ref_chosen_logps

    # rejected_ratio = log(π_θ(y_l|x) / π_ref(y_l|x))
    rejected_ratio = policy_rejected_logps - ref_rejected_logps

    # 步骤2: 计算 DPO logits（chosen 和 rejected 的奖励差）
    # logits: [batch_size]
    logits = chosen_ratio - rejected_ratio

    # 步骤3: 计算 DPO 损失
    # dpo_loss = -log(sigmoid(β * logits))
    dpo_loss = -F.logsigmoid(beta * logits).mean()

    # 步骤4: 应用标签平滑（可选）
    if label_smoothing > 0.0:
        # 标签平滑版本：混合正向和反向损失
        inverse_loss = -F.logsigmoid(-beta * logits).mean()
        dpo_loss = (1 - label_smoothing) * dpo_loss - label_smoothing * inverse_loss

    return dpo_loss
