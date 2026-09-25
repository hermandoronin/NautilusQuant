#!/usr/bin/env python3
"""Print the sign-off violations of a LibreLane run, per rule.

    python tools/signoff_summary.py flow/ihp-sg13g2/runs/<tag>

Reads the KLayout report databases (DRC, density, antenna: *.lyrdb, XML)
and the Netgen LVS report, and prints one line per violated rule with its
count. Needs only the Python standard library, so CI can run it on a
failed run and put the answer in the job log.
"""

from __future__ import annotations

import collections
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def lyrdb_counts(path: Path) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    root = ET.parse(path).getroot()
    for item in root.iter("item"):
        cat = item.findtext("category", default="?").strip().strip("'")
        counts[cat] += 1
    return counts


def main() -> int:
    run = Path(sys.argv[1])
    total = 0
    for rdb in sorted(run.glob("*-klayout-*/**/*.lyrdb")):
        counts = lyrdb_counts(rdb)
        n = sum(counts.values())
        total += n
        print(f"{rdb.relative_to(run)}: {n} item(s)")
        for cat, c in counts.most_common():
            print(f"    {c:6d}  {cat}")
    for rpt in sorted(run.glob("*-netgen-lvs/reports/lvs.netgen.rpt")):
        lines = rpt.read_text(errors="replace").splitlines()
        final = [l for l in lines if l.startswith("Final result")]
        print(f"{rpt.relative_to(run)}: {final[-1] if final else 'no final result'}")
        for l in lines:
            if "**Mismatch**" in l or "(no matching" in l:
                print("    " + l.strip())
    print(f"KLayout report items in total: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
