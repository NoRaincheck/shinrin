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
pip install shinrin[full]      # All optional dependencies
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

## Optimal Trees & Rules

CORELS (`CorelsClassifier`) and GOSDT (`GOSDTClassifier`) work out of the box
— their C++ engines are compiled into the package with bundled mini-GMP and
no TBB, so no system libraries are needed in any configuration. Install the
`[sklearn]` extra for GOSDT's binarizers and metrics; pandas is optional for
named-column output.
