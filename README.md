# DeepLearningCode

PyTorch implementations of common deep learning components, focusing on Transformer architectures and LLM training techniques.

## Contents

### Attention Mechanisms

| File | Description |
|------|-------------|
| `MultiHeadAttention.py` | Standard Multi-Head Attention |
| `GroupQueryAttention.py` | Grouped Query Attention (GQA) - reduces KV cache memory |
| `MultiLatentAttention.py` | Multi-Latent Attention (MLA) - DeepSeek style compression |
| `ScaledDotProductAttention.py` | Scaled Dot-Product Attention |

### Normalization Layers

| File | Description |
|------|-------------|
| `LayerNorm.py` | Layer Normalization |
| `RMSNorm.py` | Root Mean Square Normalization |

### Positional Encoding

| File | Description |
|------|-------------|
| `RotaryEmbedding.py` | Rotary Position Embedding (RoPE) |

### Loss Functions

| File | Description |
|------|-------------|
| `DPOLoss.py` | Direct Preference Optimization loss with label smoothing |
| `PPOLoss.py` | Proximal Policy Optimization clipped surrogate loss |
| `GRPOLoss.py` | Group Relative Policy Optimization with KL penalty |
| `SFTLoss.py` | Supervised Fine-Tuning loss (prompt masking) |
| `PretainLoss.py` | Pre-training cross-entropy loss |
| `EntropyLoss.py` | Cross-entropy and KL divergence utilities |

### Other Components

| File | Description |
|------|-------------|
| `LoRALinear.py` | Low-Rank Adaptation for parameter-efficient fine-tuning |
| `SwiGLUFFN.py` | SwiGLU Feed-Forward Network |

## Requirements

```bash
pip install torch
```

## Usage Example

```python
import torch
from MultiHeadAttention import MultiHeadAttention
from RMSNorm import RMSNorm
from SwiGLUFFN import SwiGLUFFN
from LoRALinear import LoRALinear

# Attention layer
attn = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 128, 512)  # (batch, seq_len, d_model)
out = attn(x)

# Normalization
norm = RMSNorm(d_model=512)
x_norm = norm(x)

# FFN
ffn = SwiGLUFFN(d_model=512, intermediate_dim=2048)
ffn_out = ffn(x)

# LoRA
lora = LoRALinear(in_features=512, out_features=512, rank=8)
lora_out = lora(x)
```

### Loss Functions

```python
from DPOLoss import dpo_loss
from PPOLoss import ppo_clip_loss
from GRPOLoss import grpo_loss, compute_grpo_advantages

# DPO
loss = dpo_loss(policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps, beta=0.1)

# PPO
loss = ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2)

# GRPO
advantages = compute_grpo_advantages(rewards)
loss = grpo_loss(old_log_probs, new_log_probs, advantages)
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer / MHA
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2: MLA](https://arxiv.org/abs/2405.04434)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [PPO: Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [SwiGLU: GLU Variants](https://arxiv.org/abs/2002.05202)
