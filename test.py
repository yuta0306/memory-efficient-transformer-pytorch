import torch
import torch.nn as nn

from faster_transformer.models import FasterMultiHeadAttention

if __name__ == "__main__":
    builtin = nn.MultiheadAttention(embed_dim=768, num_heads=12)
    print(builtin.in_proj_weight)

    model = FasterMultiHeadAttention(embed_dim=768, num_heads=12)
    # print(model)
    query = torch.randn(size=(4, 2048, 768), requires_grad=True)
    key = torch.randn(size=(4, 2048, 768), requires_grad=True)
    value = torch.randn(size=(4, 2048, 768), requires_grad=True)
    output = model(query, key, value)

    # print(output)
