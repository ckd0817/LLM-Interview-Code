import math
import torch


def softmax(logits):
    # compute softmax of logits
    # softmax(x_i) = exp(x_i) / sum(exp(x_j))
    # max_logits: Tensor of shape (batch_size, 1)
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    
    # Shift logits for numerical stability
    exp_shifted = torch.exp(logits - max_logits) # exp_shifted: Tensor of shape (batch_size, num_classes)
    sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True) # sum_exp: Tensor of shape (batch_size, 1)
    
    return exp_shifted / sum_exp # softmax: Tensor of shape (batch_size, num_classes)

def log_softmax(logits):
    # compute log softmax of logits
    # log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
    # max_logits: Tensor of shape (batch_size, 1)
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    
    # Log-sum-exp trick for numerical stability
    # log(sum(exp(x_j))) = max + log(sum(exp(x_j - max)))
    exp_shifted = torch.exp(logits - max_logits) # exp_shifted: Tensor of shape (batch_size, num_classes)
    log_sum_exp = max_logits + torch.log(torch.sum(exp_shifted, dim=-1, keepdim=True)) # log_sum_exp: Tensor of shape (batch_size, 1)
    return logits - log_sum_exp # log_softmax: Tensor of shape (batch_size, num_classes)

def cross_entropy_loss(logits, targets):
    """
    logits: Tensor of shape (batch_size, num_classes)
    targets: Tensor of shape (batch_size) with class indices
    """
    # 1. Compute log softmax
    log_probs = log_softmax(logits) # log_probs: Tensor of shape (batch_size, num_classes)

    # 2. Negative log likelihood loss
    batch_size = logits.size(0)
    batch_indices = torch.arange(batch_size)

    loss = -log_probs[batch_indices, targets] # loss: Tensor of shape (batch_size,)
    
    return loss.mean()

def KL_divergence(p_logits, q_logits):
    """
    p_logits: Tensor of shape (batch_size, num_classes) - logits from distribution P
    q_logits: Tensor of shape (batch_size, num_classes) - logits from distribution Q
    """
    # 1. Compute log softmax for both distributions
    p_log_softmax = log_softmax(p_logits) # p_log_softmax: Tensor of shape (batch_size, num_classes)
    q_log_softmax = log_softmax(q_logits) # q_log_softmax: Tensor of shape (batch_size, num_classes)
    
    # 2. Compute softmax for P to get probabilities
    p_softmax = softmax(p_logits) # p_softmax: Tensor of shape (batch_size, num_classes)
    
    # 3. KL Divergence D_KL(P || Q) = sum(P(x) * (log P(x) - log Q(x)))
    kl_div = torch.sum(p_softmax * (p_log_softmax - q_log_softmax), dim=-1) # kl_div: Tensor of shape (batch_size,)
    
    return kl_div.mean()
    