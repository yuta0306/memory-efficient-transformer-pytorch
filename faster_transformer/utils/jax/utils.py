from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

__all__ = [
    "dynamic_slice",
    "map_",
    "scan",
]


def dynamic_slice(x, starts, sizes):
    starts = [np.clip(starts[i], 0, x.shape[i] - sizes[i]) for i in range(len(starts))]
    for i, (start, size) in enumerate(zip(starts, sizes)):
        x = torch.index_select(
            x, i, torch.tensor(range(start, start + size), device=x.device)
        )
    return x


def map_(
    f: Callable, xs: Union[list, tuple, np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    t = [f(x) for x in xs]
    return tuple(map(torch.stack, zip(*t)))


def scan(
    f: Callable,
    init: int,
    xs: Optional[Union[list, tuple, np.ndarray, torch.Tensor]] = None,
    length: Optional[int] = None,
) -> Tuple[int, torch.Tensor]:
    if xs is None:
        if length is None:
            raise ValueError("length is expected to type int, but None is given")
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)
