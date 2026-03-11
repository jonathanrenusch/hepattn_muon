"""Load shared particle-selection defaults from ``selection_defaults.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULTS_PATH = Path(__file__).parent / "selection_defaults.yaml"


def load_selection_defaults(path: str | Path | None = None) -> dict:
    """Return the selection dict from the shared YAML file.

    Parameters
    ----------
    path : str | Path | None
        Override path.  When *None*, the ``selection_defaults.yaml`` next to
        this module is used.
    """
    p = Path(path) if path is not None else _DEFAULTS_PATH
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg["selection"]
