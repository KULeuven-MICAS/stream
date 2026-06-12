"""Matrix-form affine transform used by the base workload model.

Vendored from the SNAX-MLIR project (``snaxc.ir.dart.affine_transform``) so that
the core (non-AIE) workload path carries no dependency on ``snax-mlir`` — which is
only installable from git and is therefore not publishable to PyPI. The AIE
codegen path still depends on ``snax-mlir`` for its TSL layout machinery; only this
self-contained class is needed by the base model, so it is vendored rather than
imported.

Source: https://github.com/KULeuven-MICAS/snax-mlir
Original license: Apache License v2.0 with LLVM Exceptions.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt
from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)


@dataclass(frozen=True)
class AffineTransform:
    """
    An affine transform mirroring the functionality of xDSLs and MLIRs
    AffineMap, but represented in matrix form to make life much easier.
    This is possible if you don't have to support floordiv/ceildiv operations.
    """

    A: npt.NDArray[np.int_]  # Transformation matrix
    b: npt.NDArray[np.int_]  # Translation vector

    def __post_init__(self):
        # Validate dimensions
        if self.A.ndim != 2:  # noqa: PLR2004 -- a transformation matrix is 2-dimensional by definition
            raise ValueError("Matrix A must be 2-dimensional.")
        if self.b.ndim != 1:
            raise ValueError("Vector b must be 1-dimensional.")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("Matrix A and vector b must have compatible dimensions.")

    @classmethod
    def from_affine_map(cls, map: AffineMap) -> Self:
        """
        Return the affine transform representation of the given affine map.

        For this, the affine map must be a pure linear transformation (i.e., no floordiv/ceildiv/modulo operations)
        """

        # check for pure linear transformation
        for result in map.results:
            for expr in result.dfs():
                if isinstance(expr, AffineBinaryOpExpr):
                    if expr.kind in (
                        AffineBinaryOpKind.FloorDiv,
                        AffineBinaryOpKind.CeilDiv,
                        AffineBinaryOpKind.Mod,
                    ):
                        raise ValueError("Affine map is not a pure linear transformation")

        # generate a list with n zeros and a 1 at index d:
        # [0, 0, 0, 1]
        def generate_one_list(n: int, d: int):
            return [1 if x == d else 0 for x in range(n)]

        # determine indices of the matrices a and b by getting the unit response of every dimension

        # bias b is determined by setting all dimensions to zero
        b = np.array(map.eval(generate_one_list(map.num_dims, -1), []))

        # columns of a are determined by toggling every dimension separately
        a = np.zeros((len(map.results), map.num_dims), dtype=np.int_)
        for dim in range(map.num_dims):
            temp = np.array(map.eval(generate_one_list(map.num_dims, dim), []))
            a[:, dim] = temp - b

        return cls(a, b)

    def to_affine_map(self) -> AffineMap:
        """
        Return the xDSL AffineMap representation of this AffineTransform
        """
        results: list[AffineExpr] = []
        for result in range(self.num_results):
            expr = AffineConstantExpr(int(self.b[result]))
            for dim in range(self.num_dims):
                if self.A[result, dim] != 0:
                    expr += AffineConstantExpr(int(self.A[result, dim])) * AffineDimExpr(dim)
            results.append(expr)
        return AffineMap(self.num_dims, 0, tuple(results))

    @property
    def num_dims(self) -> int:
        return self.A.shape[1]

    @property
    def num_results(self) -> int:
        return self.A.shape[0]

    def eval(self, x: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """
        Apply the affine transformation to a vector or a set of vectors.
        """
        if x.ndim == 1:  # Single vector
            if x.shape[0] != self.A.shape[1]:
                raise ValueError("Input vector x must have a dimension matching the number of columns in A.")
            return self.A @ x + self.b
        elif x.ndim == 2:  # noqa: PLR2004 -- a batch of vectors is a 2-dimensional array
            if x.shape[1] != self.A.shape[1]:
                raise ValueError("Input vectors in batch must have a dimension matching the number of columns in A.")
            return (self.A @ x.T).T + self.b
        else:
            raise ValueError("Input x must be 1D (vector) or 2D (batch of vectors).")

    def compose(self, other: Self) -> Self:
        """
        Combine this affine transformation with another.
        The result represents the application of `other` followed by `self`.
        """
        if self.A.shape[1] != other.A.shape[0]:
            raise ValueError("Matrix dimensions of the transformations do not align for composition.")
        new_A = self.A @ other.A
        new_b = self.A @ other.b + self.b
        return type(self)(new_A, new_b)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineTransform):
            return False
        return (self.A == other.A).all() and (self.b == other.b).all()

    def __str__(self):
        return f"AffineTransform(A=\n{self.A},\nb={self.b})"
