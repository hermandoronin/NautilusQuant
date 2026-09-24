# SPDX-License-Identifier: Apache-2.0
"""NQX-S1 on wafer.space GF180MCU: chip-level cocotb runner.

Builds tb_chip (pad wrapper) + chip_top + NQX-S1 RTL, or the gate-level
netlist from final/ (GL=1), with the PDK pad models, and runs the pin-level
tests of verif/cocotb/test_top.py, including the silicon golden vectors.

    make sim        # RTL
    make sim-gl     # gate level, after a LibreLane run
    NQX_PAD_STUBS=1 python3 chip_top_tb.py   # without a PDK (behavioural pads)
"""

import os
import sys
from pathlib import Path

from cocotb_tools.runner import get_results, get_runner

sim = os.getenv("SIM", "icarus")
gl = bool(os.getenv("GL"))
stubs = bool(os.getenv("NQX_PAD_STUBS"))
pdk_root = os.getenv("PDK_ROOT", Path(__file__).resolve().parent / "../gf180mcu")
pdk = os.getenv("PDK", "gf180mcuD")
scl = os.getenv("SCL", "gf180mcu_fd_sc_mcu7t5v0")
pad = os.getenv("PAD", "gf180mcu_fd_io")
slot = os.getenv("SLOT", "0p5x0p5")

proj = Path(__file__).resolve().parent
silicon = proj / "../../.."
rtl = silicon / "rtl"

RTL_FILES = [
    "nqx_s1_top.sv", "nqx_s1_hsio.sv", "nqx_s1_core.sv", "nqx_s1_cordic.sv", "nqx_s1_cordic_iter.sv",
    "nqx_s1_rcode.sv", "nqx_s1_rsign.sv", "nqx_s1_tcode.sv", "nqx_s1_qdec.sv",
    "nqx_s1_div28.sv", "nqx_s1_rf.sv",
]
IP = ["logo", "marker", "qrcode_id", "shuttle_id", "project_id"]


def runner():
    defines = {f"SLOT_{slot.upper()}": True, f"PAD_{pad}": True, f"SCL_{scl}": True}
    if slot == "0p5x0p5":
        defines["NQX_ITER_CORDIC"] = 1
    sources = [proj / "tb_chip.sv"]
    if gl:
        lib = Path(pdk_root) / pdk / "libs.ref" / scl / "verilog"
        sources += [lib / f"{scl}.v", lib / "primitives.v", proj / "../final/pnl/chip_top.pnl.v"]
        defines.update({"FUNCTIONAL": True, "USE_POWER_PINS": True})
    else:
        sources += [proj / "../src/chip_top.sv", proj / "../src/chip_core.sv"]
        sources += [rtl / f for f in RTL_FILES]
    if stubs:
        sources.append(proj / "pad_stubs.v")
    else:
        sources.append(Path(pdk_root) / pdk / f"libs.ref/{pad}/verilog/{pad}.v")
        sources += [proj / f"../ip/gf180mcu_ws_ip__{n}/vh/gf180mcu_ws_ip__{n}.v" for n in IP]

    build_dir = proj / "sim_build" / ("gl" if gl else "rtl")
    r = get_runner(sim)
    r.build(
        sources=sources,
        hdl_toplevel="tb_chip",
        defines=defines,
        includes=[proj / "../src", rtl],
        build_args=["-g2012"] if sim == "icarus" else ["--timing", "-Wno-fatal"],
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        always=True,
    )
    # The cocotb runner exports this process's sys.path as PYTHONPATH.
    sys.path[:0] = [str(silicon / "verif/cocotb"), str(silicon / "model")]
    env = {"NQX_VECTORS": os.getenv("NQX_VECTORS", "00*" if gl else "0[0-4]*")}
    results = r.test(
        hdl_toplevel="tb_chip",
        test_module="test_top",
        test_dir=silicon / "verif/cocotb",
        build_dir=build_dir,
        extra_env=env,
    )
    total, failed = get_results(results)
    assert total > 0 and failed == 0, f"{failed} of {total} tests failed"


if __name__ == "__main__":
    runner()
