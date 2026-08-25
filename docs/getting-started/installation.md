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
pip install shinrin[pandas]    # pandas for SkopeRules / OrdtClassifier
pip install shinrin[onnx]      # ONNX export support
pip install shinrin[tabicl]    # TabICL torch backend + Hugging Face checkpoint download
pip install shinrin[mojo]      # Mojo kernels (`just build-*-mojo`)
pip install shinrin[full]      # All core optional dependencies
```

Benchmark-only extras used by `scripts/benchmarks/`:

```bash
pip install shinrin[tabm-bench]    # PyTorch reference for bench_tabm.py --with-torch
pip install shinrin[tabicl-bench]  # upstream TabICL package for bench_tabicl.py --with-upstream
```

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
- [Modular](https://www.modular.com/) / Mojo CLI (optional — only to build the
  experimental Mojo kernels via the `just build-*-mojo` recipes)

## Optimal Trees & Rules

CORELS (`CorelsClassifier`) and SPOT (`SPOTClassifier`) work out of the box
— their C++ engines are compiled into the package with bundled mini-GMP and
no TBB, so no system libraries are needed in any configuration. Install the
`[sklearn]` extra for SPOT's binarizers and metrics; pandas is optional for
named-column output.
