"""Checkpoint download and conversion for TabICL.

The Hugging Face checkpoints (``jingang/TabICL``) are Lightning-style dicts
with ``config`` and ``state_dict`` keys. The torch backend loads the raw
checkpoint; the NumPy and Mojo backends need plain arrays, so we convert the
checkpoint once into an uncompressed ``.npz`` file stored next to it:

- ``__format_version__``: integer format marker,
- ``__config__``: JSON string with the architecture hyper-parameters,
- every state-dict tensor under its original name, cast to float32.

Conversion requires ``torch``; the NumPy/Mojo backends only read the ``npz``
and raise an informative error if it has not been produced yet.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

HF_REPO_ID = "jingang/TabICL"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main"

CLASSIFIER_V2 = "tabicl-classifier-v2-20260212.ckpt"
REGRESSOR_V2 = "tabicl-regressor-v2-20260212.ckpt"

NPZ_FORMAT_VERSION = 1


def default_cache_dir() -> Path:
    """Return the default cache directory for checkpoints."""
    return Path(
        os.environ.get(
            "SHINRIN_TABICL_CACHE", str(Path.home() / ".cache" / "shinrin" / "tabicl")
        )
    )


def _download_url(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` atomically via a temporary file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "shinrin/0.1"})
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            os.fdopen(fd, "wb") as fh,
        ):
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        tmp_path.replace(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _hf_hub_download(filename: str, dest: Path) -> bool:
    """Try downloading via ``huggingface_hub``; return False if unavailable."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
    except Exception:  # noqa: BLE001 - fall back to direct download below
        return False
    path = Path(path)
    if path != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    return True


def ensure_checkpoint(
    filename: str = CLASSIFIER_V2,
    model_path: str | Path | None = None,
    allow_auto_download: bool = True,
) -> Path:
    """Locate or download a raw checkpoint and return its path.

    Parameters
    ----------
    filename : str
        Checkpoint file name in the HF repo (e.g. ``tabicl-classifier-v2-...ckpt``).
    model_path : str or Path, optional
        Directory to store the checkpoint. Defaults to the shinrin cache dir.
    allow_auto_download : bool
        Whether downloading is allowed when the checkpoint is missing.
    """
    directory = default_cache_dir() if model_path is None else Path(model_path)
    ckpt_path = directory / filename
    if ckpt_path.exists():
        return ckpt_path
    if not allow_auto_download:
        raise FileNotFoundError(
            f"Checkpoint '{filename}' not found in '{directory}' and automatic download is disabled. "
            f"Set allow_auto_download=True or pass model_path pointing at the checkpoint."
        )
    url = f"{HF_BASE_URL}/{filename}"
    try:
        _download_url(url, ckpt_path)
    except Exception:
        if ckpt_path.exists():
            ckpt_path.unlink()
        if _hf_hub_download(filename, ckpt_path):
            return ckpt_path
        raise
    return ckpt_path


def convert_checkpoint_to_npz(
    ckpt_path: str | Path, npz_path: str | Path | None = None
) -> Path:
    """Convert a raw checkpoint into an ``.npz`` archive (requires torch)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Converting a TabICL checkpoint requires torch. "
            "Install it with: uv pip install torch  (or pip install torch), then retry. "
            "Alternatively install the 'tabicl' extra: pip install 'shinrin[tabicl]'"
        ) from exc

    ckpt_path = Path(ckpt_path)
    if npz_path is None:
        npz_path = ckpt_path.with_suffix(".npz")
    npz_path = Path(npz_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint '{ckpt_path}' lacks 'config'/'state_dict' keys; not a TabICL checkpoint."
        )

    config_json = json.dumps(checkpoint["config"])
    arrays: dict[str, np.ndarray] = {
        "__format_version__": np.array(NPZ_FORMAT_VERSION),
        "__config__": np.array(config_json),
    }
    for name, tensor in checkpoint["state_dict"].items():
        arrays[name] = tensor.detach().cpu().to(torch.float32).numpy()

    npz_tmp = npz_path.with_suffix(".npz.part")
    with open(npz_tmp, "wb") as fh:
        np.savez(fh, **arrays)  # ty: ignore[invalid-argument-type]
    npz_tmp.replace(npz_path)
    return npz_path


def load_npz(npz_path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Load a converted checkpoint.

    Returns
    -------
    config : dict
        Parsed architecture metadata.
    params : dict of str -> ndarray
        State-dict tensors keyed by their original names.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        version = int(data["__format_version__"])
        if version != NPZ_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported TabICL npz format version {version} in '{npz_path}'."
            )
        config = json.loads(str(data["__config__"]))
        params = {k: data[k] for k in data.files if not k.startswith("__")}
    return config, params


def ensure_npz(
    filename: str = CLASSIFIER_V2,
    model_path: str | Path | None = None,
    allow_auto_download: bool = True,
) -> tuple[Path, dict, dict[str, np.ndarray]]:
    """Ensure a converted ``.npz`` exists and load it.

    Downloads the raw checkpoint and converts it when needed.
    """
    ckpt_path = ensure_checkpoint(
        filename=filename,
        model_path=model_path,
        allow_auto_download=allow_auto_download,
    )
    npz_path = ckpt_path.with_suffix(".npz")
    if not npz_path.exists():
        convert_checkpoint_to_npz(ckpt_path, npz_path)
    config_dict, params = load_npz(npz_path)
    return npz_path, config_dict, params
