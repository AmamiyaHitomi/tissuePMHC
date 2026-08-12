#!/usr/bin/env python3
"""Run issue-9 unit tests without requiring pytest."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parent / "tests" / "test_issue9.py"
    specification = importlib.util.spec_from_file_location("issue9_tests", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load tests from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    tests = sorted(name for name in vars(module) if name.startswith("test_"))
    for name in tests:
        getattr(module, name)()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
