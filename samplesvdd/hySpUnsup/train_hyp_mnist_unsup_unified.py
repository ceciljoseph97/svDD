"""
Unified entrypoint for MNIST unsupervised multi-sphere OCC training.

Use this script as the single stable interface.
Internally it routes to the latest adaptive trainer implementation.

Examples:
- python hySpUnsup/train_hyp_mnist_unsup_unified.py --mnist_processed_dir ... --device cuda --geometry euclidean
- python hySpUnsup/train_hyp_mnist_unsup_unified.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import train_hyp_mnist_unsup_v3 as _impl


def main() -> None:
    _impl.main()


if __name__ == "__main__":
    main()
