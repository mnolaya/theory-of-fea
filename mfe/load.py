from __future__ import annotations

from attrs import define, field
import numpy as np
import numpy.typing as npt

def 

@define
class SurfaceTraction:
    order: int
    constants: np.ndarray
    _ndim: int = 2
    funcs: tuple = field(init=False)

    def __attrs_post_init__(self) -> None:
        func_dict = {
            1: self._first_order,
            2: self._second_order,
        }
        self.funcs = tuple(func_dict[self.order] for _ in range(self._ndim))
        
    @staticmethod
    def _first_order(x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return c[1]*x + c[0]

    @staticmethod
    def _second_order(x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return c[2]*x**2 + c[1]*x + c[0]
    

    