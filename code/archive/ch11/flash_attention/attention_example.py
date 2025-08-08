import torch
import torch.nn.functional as F

# Example usage of FlashAttention via PyTorch API
# Q, K, V tensors of shape (batch, heads, seq_len, head_dim)
batch, heads, seq_len, head_dim = 1, 8, 512, 64
Q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
K = torch.randn_like(Q)
V = torch.randn_like(Q)

# Fused FlashAttention call
output = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=0.0)
print("Attention output shape:", output.shape)
