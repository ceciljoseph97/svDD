"""
Delegates to hySpCheck/visualize_inclass_extremes_panel.py (MNIST + CIFAR 2x2 panel).
Same CLI; run from anywhere:
  python visualize_inclass_extremes_panel.py --mnist_processed_dir ... --mnist_checkpoint ...
      --cifar_data_root ... --cifar_checkpoint ... --out_path ...
"""
import sys
from pathlib import Path

_HY = Path(__file__).resolve().parent.parent / "hySpCheck"
if not _HY.is_dir():
    raise FileNotFoundError(_HY)
sys.path.insert(0, str(_HY))

from visualize_inclass_extremes_panel import main  # noqa: E402

if __name__ == "__main__":
    main()
