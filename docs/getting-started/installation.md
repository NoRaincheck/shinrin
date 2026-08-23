# Installation

## From PyPI

The easiest way to install Shinrin is via pip:

```bash
pip install shinrin
```

## Optional Dependencies

Shinrin offers several optional dependency groups:

```bash
pip install shinrin[sklearn]   # scikit-learn for benchmarks and SHAP plotting
pip install shinrin[onnx]      # ONNX export support
pip install shinrin[pandas]    # pandas integration
pip install shinrin[mojo]      # CPU Mojo training kernels (TabM / MLP)
pip install shinrin[metal]     # Apple-GPU (Metal) kernels — see below
pip install shinrin[full]      # All optional dependencies
```

## Metal (Apple GPU) Backends

The TabM and MLP trainers ship experimental Apple-GPU backends written in
Mojo. In addition to the `metal` extra they require:

- Apple Silicon (M1–M4) with macOS 15 or newer
- Xcode 26 with the Metal toolchain component:

  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```

After installing, build and select the backend:

```bash
just build-tabm-metal                     # shinrin/_native_tabm_gpu.so
just build-mlp-metal                      # shinrin/_native_mlp_gpu.so
SHINRIN_TABM_BACKEND=metal python ...     # or SHINRIN_MLP_BACKEND=metal
```

These backends are experimental: on current MAX releases large training
runs can silently drop kernel launches (upstream compiler-service
instability). The NumPy and CPU Mojo paths are unaffected.

## From Source

To install from source, clone the repository and install with uv:

```bash
git clone https://github.com/NoRaincheck/shinrin.git
cd shinrin
uv sync --all-extras
```

## System Requirements

- Python 3.10 or higher
- Rust toolchain (for building from source)
