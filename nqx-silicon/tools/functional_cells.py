#!/usr/bin/env python3
"""Make a zero-delay functional copy of a foundry standard-cell Verilog library.

The IHP SG13G2 flip-flop models clock their UDPs from `delayed_*` nets that
only `$setuphold` timing checks drive. Icarus Verilog does not implement
those checks, so the flops would output X. This script removes the specify
blocks and connects every `delayed_X` net straight to `X`. Logic functions
are untouched. Timing is signed off by static timing analysis instead.

    python tools/functional_cells.py <cells.v> <out.v>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def convert(text: str) -> str:
    text = re.sub(r"\bspecify\b.*?\bendspecify\b", "", text, flags=re.S)
    # Drop the declarations of the delayed nets, then use the ports directly.
    text = re.sub(r"^\s*wire\s+delayed_\w+(\s*,\s*delayed_\w+)*\s*;\s*$", "", text, flags=re.M)
    return re.sub(r"\bdelayed_(\w+)", r"\1", text)


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(convert(src.read_text()))


if __name__ == "__main__":
    main()
