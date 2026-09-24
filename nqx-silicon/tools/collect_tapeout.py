#!/usr/bin/env python3
"""Collect the sign-off views of a LibreLane run into tapeout/<process>/.

    python tools/collect_tapeout.py flow/ihp-sg13g2/runs/<tag> --process ihp-sg13g2

Copies the final GDS (gzip), netlists, extracted SPICE (gzip), SDC, metrics,
the layout render and the sign-off reports (large ones gzipped),
writes SIGNOFF.md (a table generated from metrics.json) and SHA256SUMS.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEYS = [
    ("design__die__bbox", "Die bounding box [µm]"),
    ("design__die__area", "Die area [µm²]"),
    ("design__core__area", "Core area [µm²]"),
    ("design__instance__count", "Instances (all)"),
    ("design__instance__count__stdcell", "Standard cells"),
    ("design__instance__area__stdcell", "Standard-cell area [µm²]"),
    ("design__instance__utilization", "Core utilization"),
    ("design__io", "Top-level ports (signal and supply)"),
    ("route__wirelength", "Routed wire length [µm]"),
    ("route__drc_errors", "Detailed-routing DRC errors"),
    ("magic__drc_error__count", "Magic DRC errors"),
    ("klayout__drc_error__count", "KLayout DRC errors"),
    ("magic__illegal_overlap__count", "Magic illegal overlaps"),
    ("design__lvs_error__count", "LVS errors"),
    ("design__lvs_unmatched_device__count", "LVS unmatched devices"),
    ("design__lvs_unmatched_net__count", "LVS unmatched nets"),
    ("design__lvs_unmatched_pin__count", "LVS unmatched pins"),
    ("antenna__violating__nets", "Antenna-violating nets (OpenROAD)"),
    ("antenna__violating__pins", "Antenna-violating pins (OpenROAD)"),
    ("klayout__antenna_error__count", "KLayout antenna errors"),
    ("klayout__density_error__count", "KLayout density errors"),
    ("design__xor_difference__count", "Magic/KLayout GDS XOR differences"),
    ("timing__setup__ws", "Worst setup slack, all corners [ns]"),
    ("timing__hold__ws", "Worst hold slack, all corners [ns]"),
    ("timing__setup_vio__count", "Setup violations"),
    ("timing__hold_vio__count", "Hold violations"),
    ("design__max_slew_violation__count", "Max slew violations"),
    ("design__max_cap_violation__count", "Max capacitance violations"),
    ("design__max_fanout_violation__count", "Nets above the fanout guideline (informational)"),
    ("clock__skew__worst_setup", "Clock skew (setup) [ns]"),
    ("power__total", "Total power, typical corner [W]"),
    ("power__leakage__total", "Leakage power [W]"),
    ("ir__drop__worst", "Worst IR drop [V]"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return " ".join(fmt(x) for x in v)
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--process", default="ihp-sg13g2")
    ap.add_argument("--top", default="chip_top")
    args = ap.parse_args()

    final = args.run / "final"
    out = ROOT / "tapeout" / args.process
    if out.exists():
        shutil.rmtree(out)
    (out / "reports").mkdir(parents=True)

    gds = final / "gds" / f"{args.top}.gds"
    with open(gds, "rb") as src, gzip.open(out / f"{args.top}.gds.gz", "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    # Netlists (logical and powered) and the constraints; LibreLane saves them
    # as final/<view>/<top>.<view>.<ext>.
    for view in ("nl", "pnl", "sdc"):
        for f in sorted((final / view).glob(f"{args.top}.*")):
            shutil.copy(f, out / f.name)
    # Layout-extracted SPICE netlist (what LVS compared) and the layout render.
    spice = final / "spice" / f"{args.top}.spice"
    if spice.is_file():
        with open(spice, "rb") as src, gzip.open(out / f"{args.top}.spice.gz", "wb", compresslevel=9) as dst:
            shutil.copyfileobj(src, dst)
    render = final / "render" / f"{args.top}.png"
    if render.is_file():
        shutil.copy(render, out / render.name)
    metrics = json.loads((final / "metrics.json").read_text())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True))

    for pattern in [
        "*-magic-drc/reports/*",
        "*-klayout-drc/reports/*",
        "*-netgen-lvs/reports/*",
        "*-klayout-antenna/reports/*",
        "*-klayout-density/reports/*",
        "*-openroad-stapostpnr/summary.rpt",
        "*-openroad-stapostpnr/*/max.rpt",
        "*-openroad-stapostpnr/*/min.rpt",
        "*-openroad-stapostpnr/*/checks.rpt",
        "*-openroad-stapostpnr/*/power.rpt",
        "*-openroad-irdropreport/*.rpt",
        "*-openroad-checkantennas*/reports/*",
        "*-misc-reportmanufacturability/*.rpt",
    ]:
        for f in sorted(args.run.glob(pattern)):
            if not f.is_file():
                continue
            dst = out / "reports" / str(f.relative_to(args.run)).replace("/", "__")
            if f.stat().st_size > 1024 * 1024:           # path listings: keep, compressed
                with open(f, "rb") as src, gzip.open(f"{dst}.gz", "wb", compresslevel=9) as gz:
                    shutil.copyfileobj(src, gz)
            else:
                shutil.copy(f, dst)

    # Last commit that changed a design input (RTL, flow configuration,
    # constraints, macros, scripts); simulation files and runs do not count.
    flow = args.run.resolve().parents[1].relative_to(ROOT)
    rev = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "rtl", str(flow),
         f":(exclude){flow}/sim", f":(exclude){flow}/runs"],
        cwd=ROOT, capture_output=True, text=True)
    lines = [
        f"# Sign-off summary: NQX-S1, {args.process}",
        "",
        f"Generated by `tools/collect_tapeout.py` from LibreLane run `{args.run.name}`"
        f" (design sources at commit `{rev.stdout.strip() or 'unknown'}`,"
        " the last change to `rtl/` or the flow configuration).",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, label in KEYS:
        if key in metrics:
            lines.append(f"| {label} | {fmt(metrics[key])} |")
    corners = sorted(
        {k.split("corner:")[1] for k in metrics if "timing__setup__ws__corner:" in k}
    )
    if corners:
        lines += ["", "| Corner | Setup WS [ns] | Setup TNS [ns] | Hold WS [ns] | Hold TNS [ns] |",
                  "|---|---|---|---|---|"]
        for c in corners:
            g = lambda m: fmt(metrics.get(f"timing__{m}__corner:{c}", "n/a"))  # noqa: E731
            lines.append(f"| {c} | {g('setup__ws')} | {g('setup__tns')} | {g('hold__ws')} | {g('hold__tns')} |")
    (out / "SIGNOFF.md").write_text("\n".join(lines) + "\n")

    sums = []
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.name != "SHA256SUMS":
            sums.append(f"{sha256(f)}  {f.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print((out / "SIGNOFF.md").read_text())


if __name__ == "__main__":
    main()
