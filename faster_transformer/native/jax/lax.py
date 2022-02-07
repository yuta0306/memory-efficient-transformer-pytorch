import warnings
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import torch

from . import core
from ._src import dtypes

Array = Any
DType = torch.dtype


def convert_element_type(operand: Array, new_dtype: DType) -> Array:
    """Elementwise cast.

    Wraps XLA's `ConvertElementType
    <https://www.tensorflow.org/xla/operation_semantics#convertelementtype>`_
    operator, which performs an elementwise conversion from one type to another.
    Similar to a C++ `static_cast`.

    Args:
      operand: an array or scalar value to be cast
      new_dtype: a NumPy dtype representing the target type.

    Returns:
      An array with the same shape as `operand`, cast elementwise to `new_dtype`.
    """
    if hasattr(operand, "__jax_array__"):
        operand = operand.__jax_array__()
    return _convert_element_type(operand, new_dtype, weak_type=False)


def _convert_element_type(
    operand: Array, new_dtype: Optional[DType] = None, weak_type: bool = False
):
    # Don't canonicalize old_dtype because x64 context might cause
    # un-canonicalized operands to be passed in.
    old_dtype = dtypes.dtype(operand, canonicalize=False)
    old_weak_type = dtypes.is_weakly_typed(operand)

    if new_dtype is None:
        new_dtype = old_dtype

    new_dtype = dtypes.dtype(new_dtype, canonicalize=True)
    new_weak_type = bool(weak_type)

    if dtypes.issubdtype(old_dtype, torch.complex128) and not dtypes.issubdtype(
        new_dtype, torch.complex128
    ):
        msg = "Casting complex values to real discards the imaginary part"
        warnings.warn(msg, UserWarning, stacklevel=2)

    # Python has big integers, but convert_element_type(2 ** 100, np.float32) need
    # not be an error since the target dtype fits the value. Handle this case by
    # converting to a NumPy array before calling bind. Without this step, we'd
    # first canonicalize the input to a value of dtype int32 or int64, leading to
    # an overflow error.
    if type(operand) is int:
        operand = torch.as_tensor(operand, new_dtype)
        old_weak_type = False

    if (old_dtype, old_weak_type) == (new_dtype, new_weak_type) and isinstance(
        operand, (core.Tracer, device_array.DeviceArray)
    ):
        return operand
    else:
        return convert_element_type_p.bind(
            operand, new_dtype=new_dtype, weak_type=new_weak_type
        )


def _dynamic_slice_indices(operand: torch.Tensor, start_indices: Any):
    # Normalize the start_indices w.r.t. operand.shape
    if len(start_indices) != operand.ndim:
        msg = (
            "Length of slice indices must match number of operand dimensions ({} "
            "vs {})"
        )
        raise ValueError(msg.format(len(start_indices), operand.shape))
    if not isinstance(start_indices, (tuple, list)):
        if start_indices.ndim != 1:
            raise ValueError(
                "Slice indices must be a 1D sequence, got {}".format(
                    start_indices.shape
                )
            )
        start_indices = [i for i in start_indices]
    return [
        torch.as_tensor(i + d if i < 0 else i, torch.long)
        if isinstance(i, (int,)) and core.is_constant_dim(d)
        else lax.select(
            lax.lt(i, lax._const(i, 0)),
            lax.add(
                i, lax.convert_element_type(core.dimension_as_value(d), lax._dtype(i))
            ),
            i,
        )
        for i, d in zip(start_indices, operand.shape)
    ]


def dynamic_slice(
    operand: Any,
    start_indices: Sequence[Any],
    slice_sizes: Sequence[Union[int, Any]],
):
    pass


def map_(f: Callable, xs: Iterable) -> tuple:
    print(xs)
    for i, x in enumerate(xs):
        print(i, x)
    t = [f(x) for x in xs]
    # print("t >>>", t)
    # print("t.zip >>>", tuple(zip(*t)))
    return tuple(map(torch.stack, zip(*t)))


def scan(
    f: Callable,
    init: Any,
    xs: Optional[Iterable] = None,
    length: Optional[int] = None,
):
    assert length is not None and xs is None
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)
