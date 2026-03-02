"""
熵相关损失函数

包含用于大语言模型训练的常用损失函数：
- Softmax：数值稳定的 softmax 实现
- Log Softmax：数值稳定的 log softmax 实现
- Cross Entropy Loss：交叉熵损失
- KL Divergence：KL 散度（用于知识蒸馏、DPO 等）
"""

import math
import torch


def softmax(logits):
    """
    数值稳定的 Softmax 函数

    通过减去最大值来避免数值溢出问题。

    Args:
        logits: 未归一化的对数概率 [batch_size, num_classes]

    Returns:
        softmax: 归一化的概率分布 [batch_size, num_classes]
    """
    # 步骤1: 计算每个样本的最大值用于数值稳定性
    # max_logits: [batch_size, 1]
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)

    # 步骤2: 减去最大值后计算 exp，避免溢出
    # exp_shifted: [batch_size, num_classes]
    exp_shifted = torch.exp(logits - max_logits)

    # 步骤3: 计算 softmax 分母
    # sum_exp: [batch_size, 1]
    sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)

    # 步骤4: 归一化得到概率分布
    # softmax: [batch_size, num_classes]
    return exp_shifted / sum_exp


def log_softmax(logits):
    """
    数值稳定的 Log Softmax 函数

    使用 Log-Sum-Exp 技巧保证数值稳定性。
    公式: log_softmax(x) = x - log(sum(exp(x)))

    Args:
        logits: 未归一化的对数概率 [batch_size, num_classes]

    Returns:
        log_softmax: 对数概率 [batch_size, num_classes]
    """
    # 步骤1: 计算最大值用于数值稳定性
    # max_logits: [batch_size, 1]
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)

    # 步骤2: Log-Sum-Exp 技巧
    # log(sum(exp(x))) = max + log(sum(exp(x - max)))
    # exp_shifted: [batch_size, num_classes]
    exp_shifted = torch.exp(logits - max_logits)

    # log_sum_exp: [batch_size, 1]
    log_sum_exp = max_logits + torch.log(torch.sum(exp_shifted, dim=-1, keepdim=True))

    # 步骤3: 计算 log softmax
    # log_softmax: [batch_size, num_classes]
    return logits - log_sum_exp


def cross_entropy_loss(logits, targets):
    """
    交叉熵损失函数

    用于分类任务的标准损失函数。
    公式: CE = -log(p[y])，其中 y 是真实类别

    Args:
        logits: 模型输出的未归一化对数概率 [batch_size, num_classes]
        targets: 真实类别索引 [batch_size]

    Returns:
        loss: 标量损失值
    """
    # 步骤1: 计算 log softmax
    # log_probs: [batch_size, num_classes]
    log_probs = log_softmax(logits)

    # 步骤2: 提取真实类别的 log 概率
    batch_size = logits.size(0)
    batch_indices = torch.arange(batch_size)

    # loss: [batch_size]
    loss = -log_probs[batch_indices, targets]

    # 步骤3: 返回平均损失
    return loss.mean()


def KL_divergence(p_logits, q_logits):
    """
    KL 散度（Kullback-Leibler Divergence）

    衡量两个概率分布 P 和 Q 之间的差异。
    公式: D_KL(P || Q) = sum(P(x) * (log P(x) - log Q(x)))

    常用于知识蒸馏、DPO 训练等场景。

    Args:
        p_logits: 分布 P 的 logits [batch_size, num_classes]
        q_logits: 分布 Q 的 logits [batch_size, num_classes]

    Returns:
        kl_div: 标量 KL 散度值
    """
    # 步骤1: 计算两个分布的 log softmax
    # p_log_softmax: [batch_size, num_classes]
    # q_log_softmax: [batch_size, num_classes]
    p_log_softmax = log_softmax(p_logits)
    q_log_softmax = log_softmax(q_logits)

    # 步骤2: 计算 P 的概率分布
    # p_softmax: [batch_size, num_classes]
    p_softmax = softmax(p_logits)

    # 步骤3: 计算 KL 散度
    # D_KL(P || Q) = sum(P(x) * (log P(x) - log Q(x)))
    # kl_div: [batch_size]
    kl_div = torch.sum(p_softmax * (p_log_softmax - q_log_softmax), dim=-1)

    return kl_div.mean()
