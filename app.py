from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_pulse_tracker.dashboard import render_dashboard


if __name__ == "__main__":
    render_dashboard()
