"""Python 3.13 compatibility launcher for MHCflurry 2.2.1 downloads.

MHCflurry 2.2.1's downloads command still imports ``pipes.quote``, although
the deprecated stdlib ``pipes`` module was removed in Python 3.13.  The
equivalent supported implementation is ``shlex.quote``.  This launcher
provides only that compatibility symbol before importing the official CLI.
"""

from __future__ import annotations

import shlex
import sys
import types


compat = types.ModuleType("pipes")
compat.quote = shlex.quote
sys.modules.setdefault("pipes", compat)

from mhcflurry.downloads_command import run  # noqa: E402


if __name__ == "__main__":
    run()
