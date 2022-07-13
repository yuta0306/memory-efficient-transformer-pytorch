import pytest
import torch
import torch.nn as nn
from faster_transformer import __version__
from faster_transformer.models import FasterMultiHeadAttention, FasterTransformer
from faster_transformer.utils.stub.load_state_dict import load_state_dict


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


# @pytest.mark.skip()
def test_attention():
    builtin, model = build_models()

    query = torch.randn(size=(2, 2048, 768), requires_grad=True)
    key = torch.randn(size=(2, 2048, 768), requires_grad=True)
    value = torch.randn(size=(2, 2048, 768), requires_grad=True)

    output, attention_weights = model(query, key, value)
    output_builtin, attention_weights_builtin = builtin(query, key, value)

    assert output.size() == output_builtin.size()
    assert attention_weights is None

    eps = 1e-6
    assert nn.MSELoss()(output, output_builtin) < eps


# @pytest.mark.skip()
def test_attention_with_mask():
    builtin, model = build_models()

    query = torch.randn(size=(2, 2048, 768), requires_grad=True)
    key = torch.randn(size=(2, 2048, 768), requires_grad=True)
    value = torch.randn(size=(2, 2048, 768), requires_grad=True)
    mask = torch.zeros((2, 2048))

    output, attention_weights = model(query, key, value, mask)
    output_builtin, attention_weights_builtin = builtin(query, key, value, mask)

    assert output.size() == output_builtin.size()
    assert attention_weights is None

    eps = 1e-6
    assert nn.MSELoss()(output, output_builtin) < eps


def test_transformer():
    builtin = nn.Transformer(batch_first=True, nhead=1)
    model = FasterTransformer(activation="relu", nheads=1)
    state_dict = builtin.state_dict()

    model = load_state_dict(model, state_dict)

    src = torch.randn(size=(2, 512, 512), requires_grad=True)
    tgt = torch.randn(size=(2, 1, 512), requires_grad=True)
    output_builtin = builtin(src, tgt)
    print(output_builtin)
    output = model(src, tgt)
    print(output)

    assert output.size() == output_builtin.size()

    eps = 1e-6
    assert nn.MSELoss()(output, output_builtin) < eps
