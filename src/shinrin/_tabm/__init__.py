"""Vendored TabM core (NumPy + optional Mojo kernels).

Adapted from chapman73/tabm-lightning and yandex-research/tabm
(Apache-2.0); see NOTICE. The public scikit-learn interface lives in
:mod:`shinrin.tabm`.
"""

from shinrin._tabm._backend import get_tabm_backend, get_tabm_native

__all__ = ["get_tabm_backend", "get_tabm_native"]
