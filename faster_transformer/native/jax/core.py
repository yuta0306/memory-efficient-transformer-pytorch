import operator
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np

DimSize = Union[int, Any]  # extensible
Shape = Sequence[DimSize]


class InconclusiveDimensionOperation(Exception):
    """Raised when we cannot conclusively compute with symbolic dimensions."""

    pass


class DimensionHandler:
    """Operations on dimension sizes.

    Dimension sizes are normally integer constants, but can also be symbolic,
    e.g., masking.Poly or jax2tf.shape_poly.DimVar.

    The base class works for integers only. Subclasses are invoked when at least
    one of the operands has a type registered in _SPECIAL_DIMENSION_HANDLERS. In
    that case, all operands are guaranteed to be either the special dimension
    type, or Python integer scalars.

    Subclasses should raise InconclusiveDimensionOperation if the result cannot
    be computed in some contexts.
    """

    def is_constant(self, d: DimSize) -> bool:
        """The dimension is a constant."""
        return True

    def symbolic_equal(self, d1: DimSize, d2: DimSize) -> bool:
        """True iff the dimension sizes are equal in all contexts; False otherwise.
        Unlike `d1 == d2` this never raises InconclusiveDimensionOperation.
        """
        return d1 == d2

    def greater_equal(self, d1: DimSize, d2: DimSize) -> bool:
        """Computes `d1 >= d2`.
        Raise InconclusiveDimensionOperation if the result is different in
        different contexts.
        """
        return d1 >= d2

    def sum(self, *ds: DimSize) -> DimSize:
        """Sum of dimensions.
        Raises InconclusiveDimensionOperation if the result cannot be represented
        by the same DimSize in all contexts.
        """
        return sum(ds)

    def diff(self, d1: DimSize, d2: DimSize) -> DimSize:
        """Difference of dimensions.
        Raises InconclusiveDimensionOperation if the result cannot be represented
        by the same DimSize in all contexts.
        """
        return d1 - d2

    def divide_shape_sizes(self, s1: Shape, s2: Shape) -> DimSize:
        """Computes integer "i" such that i  * size(s2) == size(s1).

        Raise InconclusiveDimensionOperation if there is no such integer for all
        contexts,
        """
        sz1 = int(np.prod(s1))
        sz2 = int(np.prod(s2))
        if sz1 == 0 and sz2 == 0:
            return 1
        if sz1 % sz2:
            raise InconclusiveDimensionOperation(
                f"Cannot divide evenly the sizes of shapes {tuple(s1)} and {tuple(s2)}"
            )
        return sz1 // sz2

    def stride(
        self, d: DimSize, window_size: DimSize, window_stride: DimSize
    ) -> DimSize:
        """(d - window_size) // window_stride + 1"""
        return (d - window_size) // window_stride + 1

    def dilate(self, d: DimSize, dilation: int) -> DimSize:
        """Implements `0 if d == 0 else 1 + dilation * (d - 1))`"""
        return 0 if d == 0 else 1 + dilation * (d - 1)

    def as_value(self, d: DimSize):
        """Turns a dimension size into a JAX value that we can compute with."""
        return d


_dimension_handler_int = DimensionHandler()
_SPECIAL_DIMENSION_HANDLERS: Dict[type, DimensionHandler] = {}


def _dim_handler_and_canonical(
    *dlist: DimSize,
) -> Tuple[DimensionHandler, Tuple[DimSize, ...]]:
    """Finds the handler for the given dimensions; also returns the canonical dimensions.

    A dimension is canonical if it is a Python integer scalar, or has a type
    registered in _SPECIAL_DIMENSION_HANDLERS.
    """
    special_handlers = set()
    canonical = []
    for d in dlist:
        handler = _SPECIAL_DIMENSION_HANDLERS.get(type(d))
        if handler:
            special_handlers.add(handler)
            canonical.append(d)
        else:
            try:
                canonical.append(operator.index(d))
            except TypeError:
                raise TypeError(dlist)

    if len(special_handlers) > 1:
        msg = f"Dimension size operation involves multiple special dimension types {dlist}"
        raise ValueError(msg)
    return next(iter(special_handlers), _dimension_handler_int), tuple(canonical)


def is_constant_dim(d: DimSize) -> bool:
    handler, ds = _dim_handler_and_canonical(d)
    return handler.is_constant(*ds)
