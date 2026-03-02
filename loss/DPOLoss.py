import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             ref_chosen_logps, ref_rejected_logps, 
             beta=0.1, label_smoothing=0.0
             ):
    """
    Args:
        policy_chosen_logps: Log probabilities of the chosen actions under the policy model.
        policy_rejected_logps: Log probabilities of the rejected actions under the policy model.
        ref_chosen_logps: Log probabilities of the chosen actions under the reference model.
        ref_rejected_logps: Log probabilities of the rejected actions under the reference model.
        beta: KL scaling factor.
    """
    
    # 1. Compute the log ratio 
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    
    # 2. Compute the DPO logits
    logits = chosen_ratio - rejected_ratio

    # 3. Compute the DPO loss
    dpo_loss = -F.logsigmoid(beta * logits).mean()
    
    # 4. Apply label smoothing if specified
    if label_smoothing > 0.0:
        inverse_loss = -F.logsigmoid(-beta * logits).mean()
        dpo_loss = (1 - label_smoothing) * dpo_loss - label_smoothing * inverse_loss
    
    return dpo_loss
    