from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from faster_transformer import __version__
from faster_transformer.models import FasterMultiHeadAttention, FasterTransformer


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


@pytest.mark.skip()
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


@pytest.mark.skip()
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
    builtin = nn.Transformer()
    print(builtin)
    state_dict = builtin.state_dict()
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        if "in_proj_weight" in k:
            base = k.replace(".in_proj_weight", "")
            Wq, Wk, Wv = v.chunk(3, 0)
            state_dict_new[f"{base}.q_proj.weight"] = Wq
            state_dict_new[f"{base}.k_proj.weight"] = Wk
            state_dict_new[f"{base}.v_proj.weight"] = Wv
        elif "in_proj_bias" in k:
            base = k.replace(".in_proj_bias", "")
            bias_q, bias_k, bias_v = v.chunk(3, 0)
            state_dict_new[f"{base}.q_proj.bais"] = bias_q
            state_dict_new[f"{base}.k_proj.bais"] = bias_k
            state_dict_new[f"{base}.v_proj.bais"] = bias_v
        else:
            state_dict_new[k] = v
    model = FasterTransformer(activation="relu")
    print(model)
    keys = [n for n, p in model.named_parameters()]
    for key in state_dict_new.keys():
        if key not in keys:
            print(key)
    print()
    for key in keys:
        if key not in state_dict_new.keys():
            print(key)
    model.load_state_dict(state_dict_new, strict=True)
