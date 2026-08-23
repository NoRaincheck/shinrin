"""Safety net so a plain test run can never wedge the machine.

Two guards run at import time, before any test executes:

- Experimental Metal (GPU) backends are disabled unless explicitly opted in
  with ``SHINRIN_TEST_METAL=1``. A misbehaving GPU kernel can freeze the
  whole OS (the GPU warmup path already fights dropped launches and XPC
  errors), which is unacceptable for an ordinary ``pytest`` invocation.
  The opt-in keeps ``just test-tabm-metal`` / ``just test-mlp-metal``
  working unchanged.

- Native trainer thread pools default to two workers during tests via
  ``SHINRIN_TABM_THREADS`` / ``SHINRIN_MLP_THREADS``. The Mojo kernels
  otherwise spawn one thread per performance core per epoch, starving
  everything else on the machine while the suite runs.

Explicit user overrides of any of these variables always win; the Metal
opt-in is honored verbatim.
"""

from __future__ import annotations

import os
import sys

_BACKEND_VARS = ("SHINRIN_TABM_BACKEND", "SHINRIN_MLP_BACKEND")
_THREAD_VARS = ("SHINRIN_TABM_THREADS", "SHINRIN_MLP_THREADS")


def _harden_env() -> None:
    if os.environ.get("SHINRIN_TEST_METAL", "").strip() != "1":
        for var in _BACKEND_VARS:
            if os.environ.get(var, "").strip().lower() == "metal":
                del os.environ[var]
                print(
                    f"conftest: {var}=metal ignored in tests "
                    "(set SHINRIN_TEST_METAL=1 to allow)",
                    file=sys.stderr,
                )
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "2")


_harden_env()
