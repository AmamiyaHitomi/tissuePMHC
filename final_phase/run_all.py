#!/usr/bin/env python3
"""Run final-phase analyses 01-09 in dependency order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    scripts = sorted(HERE.glob("[0-9][0-9]_*.py"))
    for script in scripts:
        print(f"\n=== {script.name} ===", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
