import torch
from faster_transformer import FasterMultiHeadAttention, __version__


def test_version():
    assert __version__ == "0.1.0"


def test_attention():
    model = FasterMultiHeadAttention()
    print(model)
    query = torch.randn(size=(1, 6, 512), requires_grad=True)
    key = torch.randn(size=(1, 6, 512), requires_grad=True)
    value = torch.randn(size=(1, 6, 512), requires_grad=True)
    output = model(query, key, value)
    print(output)
