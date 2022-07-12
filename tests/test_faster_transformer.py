import torch
import torch.nn as nn
from faster_transformer import FasterMultiHeadAttention, __version__


def build_models():
    builtin = nn.MultiheadAttention(
        embed_dim=768, num_heads=12, bias=False, add_bias_kv=False, batch_first=True
    )
    Wq, Wk, Wv = builtin.in_proj_weight.chunk(3)
    Wo = builtin.out_proj.weight
    o_bias = builtin.out_proj.bias

    model = FasterMultiHeadAttention(
        embed_dim=768,
        num_heads=12,
        bias=False,
        add_bias_kv=False,
        return_attention_weights=True,
    )

    model.q_proj.weight = nn.parameter.Parameter(Wq)
    model.k_proj.weight = nn.parameter.Parameter(Wk)
    model.v_proj.weight = nn.parameter.Parameter(Wv)
    model.out_proj.weight = nn.parameter.Parameter(Wo)
    model.out_proj.bias = o_bias

    return builtin, model


def test_version():
    assert __version__ == "0.1.0"


def test_attention():
    builtin, model = build_models()

    query = torch.randn(size=(2, 4096, 768), requires_grad=True)
    key = torch.randn(size=(2, 4096, 768), requires_grad=True)
    value = torch.randn(size=(2, 4096, 768), requires_grad=True)

    output, attention_weights = model(query, key, value)
    output_builtin, attention_weights_builtin = builtin(query, key, value)

    assert output.size() == output_builtin.size()
    assert attention_weights is None

    eps = 1e-6
    assert nn.MSELoss()(output, output_builtin) < eps
