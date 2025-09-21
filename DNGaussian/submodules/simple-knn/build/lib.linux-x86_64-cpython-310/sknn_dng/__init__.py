# Lightweight wrapper that re-exports the compiled ops
from ._C import distCUDA2  # adjust names if you expose more symbols

__all__ = ["distCUDA2"]
