"""I/O utilities for loading tracking data files.

The loader is robust to simple text headers and extracts two-column numeric
X, Y data. Returns a NumPy array of shape (N, 2) in float64.
"""
from __future__ import annotations
import typing as _t
import numpy as np


def load_two_column_file(path: str) -> np.ndarray:
    """Load a two-column whitespace- or tab-separated file, skipping headers.

    The function reads lines and tries to parse two floats per line. Lines that
    can't be parsed are skipped (useful for header lines).

    Returns:
        arr: np.ndarray shape (N, 2)
    """
    data: _t.List[_t.Tuple[float, float]] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Split by whitespace or comma
            parts = line.replace(',', ' ').split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                # skip header lines or malformed rows
                continue
            data.append((x, y))

    if not data:
        return np.empty((0, 2), dtype=float)
    return np.array(data, dtype=float)


if __name__ == "__main__":
    # quick manual test
    import sys
    if len(sys.argv) > 1:
        arr = load_two_column_file(sys.argv[1])
        print(f"Loaded {arr.shape[0]} rows")