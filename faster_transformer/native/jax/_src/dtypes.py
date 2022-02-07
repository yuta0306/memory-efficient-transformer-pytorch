from typing import Any, Union

import torch

python_scalar_dtypes: dict = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float64,
    complex: torch.complex128,
}
_weak_types = [
    int,
    float,
    complex,
]


def dtype(x: Union[type, torch.Tensor, Any], *, canonicalize=False) -> torch.dtype:
    """Return the dtype object for a value or type, optionally canonicalized based on X64 mode."""
    if x is None:
        raise ValueError(f"Invalid argument to dtype: {x}.")
    elif isinstance(x, type) and x in python_scalar_dtypes:
        dt = python_scalar_dtypes[x]
    elif type(x) in python_scalar_dtypes:
        dt = python_scalar_dtypes[type(x)]
    elif isinstance(x, torch.Tensor):
        dt = torch.result_type(x, x)
    else:
        raise ValueError
    return dt


def is_weakly_typed(x):
    return type(x) in _weak_types


def _issubclass(a, b):
    """Determines if ``a`` is a subclass of ``b``.

    Similar to issubclass, but returns False instead of an exception if `a` is not
    a class.
    """
    try:
        return issubclass(a, b)
    except TypeError:
        return False


def issubdtype(a, b):
    if a == "bfloat16":
        a = torch.bfloat16
    if a == torch.bfloat16:
        if isinstance(b, torch.dtype):
            return b == torch.bfloat16
        else:
            return b in [torch.bfloat16, torch.float, torch.int]
    # if not _issubclass(b, np.generic):
    # Workaround for JAX scalar types. NumPy's issubdtype has a backward
    # compatibility behavior for the second argument of issubdtype that
    # interacts badly with JAX's custom scalar types. As a workaround,
    # explicitly cast the second argument to a NumPy type object.
    # b = np.dtype(b).type
    return a == b
