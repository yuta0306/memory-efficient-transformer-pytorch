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


@pytest.mark.skip()
def test_version():
    assert __version__ == "0.1.0"


@pytest.mark.skip()
def test_self_attention():
    builtin, model = build_models()

    query = torch.randn(size=(2, 1024, 768), requires_grad=True)

    output, attention_weights = model(query, query, query)
    output_builtin, attention_weights_builtin = builtin(query, query, query)

    assert output.size() == output_builtin.size()
    assert attention_weights is None

    eps = 1e-5
    assert nn.MSELoss()(output, output_builtin) < eps


@pytest.mark.skip()
def test_self_attention_with_long_input():
    builtin, model = build_models()

    query = torch.randn(size=(2, 4096, 768), requires_grad=True)

    output, attention_weights = model(query, query, query)
    output_builtin, attention_weights_builtin = builtin(query, query, query)

    assert output.size() == output_builtin.size()
    assert attention_weights is None

    eps = 1e-5
    assert nn.MSELoss()(output, output_builtin) < eps


@pytest.mark.skip()
def test_self_attention_with_mask():
    builtin, model = build_models()

    query = torch.randn(size=(4, 1024, 768), requires_grad=True)
    mask = torch.ones((4, 1024))
    mask_builtin = torch.ones((4, 1024))
    mask[..., -128:] = 0
    mask_builtin[..., -128:] = 0

    output, attention_weights = model(query, query, query, mask)
    output_builtin, attention_weights_builtin = builtin(
        query, query, query, mask_builtin
    )

    assert output.size() == output_builtin.size()
    assert attention_weights is None

    eps = 1e-5
    assert nn.MSELoss()(output, output_builtin) < eps


@pytest.mark.skip()
def test_transformer():
    builtin = nn.Transformer(batch_first=True, nhead=1, d_model=768)
    model = FasterTransformer(activation="relu", nheads=1, d_model=768)
    state_dict = builtin.state_dict()

    model = load_state_dict(model, state_dict)

    builtin.eval()
    model.eval()

    src = torch.randn(size=(2, 1024, 768), requires_grad=True)
    tgt = torch.randn(size=(2, 1, 768), requires_grad=True)
    output_builtin = builtin(src, tgt)
    output = model(src, tgt)

    assert output.size() == output_builtin.size()

    eps = 1e-5
    assert nn.MSELoss()(output, output_builtin) < eps


def test_transformer_with_mask():
    builtin = nn.Transformer(batch_first=True, nhead=1, d_model=768)
    model = FasterTransformer(activation="relu", nheads=1, d_model=768)
    state_dict = builtin.state_dict()

    model = load_state_dict(model, state_dict)

    builtin.eval()
    model.eval()

    src = torch.randn(size=(2, 1024, 768), requires_grad=True)
    tgt = torch.randn(size=(2, 1, 768), requires_grad=True)
    src_mask = torch.ones((2, 1024, 1024))
    src_mask[:, -256:, -256:] = 0
    output_builtin = builtin(src, tgt, src_mask)
    output = model(src, tgt, src_mask)

    assert output.size() == output_builtin.size()

    eps = 1e-5
    assert nn.MSELoss()(output, output_builtin) < eps
