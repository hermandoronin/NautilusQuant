"""pytest entry point for the cocotb RTL regressions.

    pytest verif/test_rtl.py                     # Icarus Verilog, default
    NQX_SIM=verilator pytest verif/test_rtl.py   # Verilator >= 5.036
    NQX_SCALE=5 NQX_SEED=7 pytest verif/test_rtl.py -k core   # longer / other seed
    NQX_IHP_PDK=/path/ihp-sg13g2 pytest verif/test_rtl.py -k chip_ihp   # with pad cells
    NQX_VECTORS='0[0-4]*'                        # subset of the golden-vector files
    NQX_ITER=1 pytest verif/test_rtl.py -k "core or top"   # iterative-CORDIC build
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from cocotb_tools.runner import get_results, get_runner

ROOT = Path(__file__).resolve().parents[1]
RTL = Path(os.environ.get("NQX_RTL_DIR", ROOT / "rtl"))
TB = ROOT / "verif" / "cocotb"
MODEL = ROOT / "model"
SIM = os.environ.get("NQX_SIM", "icarus")

CORE_SOURCES = [
    "nqx_s1_cordic.sv",
    "nqx_s1_cordic_iter.sv",
    "nqx_s1_rcode.sv",
    "nqx_s1_rsign.sv",
    "nqx_s1_tcode.sv",
    "nqx_s1_qdec.sv",
    "nqx_s1_div28.sv",
    "nqx_s1_rf.sv",
    "nqx_s1_core.sv",
]
TOP_SOURCES = CORE_SOURCES + ["nqx_s1_hsio.sv", "nqx_s1_top.sv"]


IHP_PDK = Path(os.environ.get("NQX_IHP_PDK", Path.home() / ".ciel" / "ihp-sg13g2"))
IHP_FLOW = ROOT / "flow" / "ihp-sg13g2"
IHP_NETLIST = Path(os.environ.get("NQX_GL_NETLIST", ROOT / "tapeout" / "ihp-sg13g2" / "chip_top.nl.v"))


def _run(
    toplevel: str,
    sources: list[Path],
    module: str,
    tag: str = "",
    test_dir: Path = TB,
    parameters: dict | None = None,
) -> None:
    iter_cordic = os.environ.get("NQX_ITER", "0")
    name = f"{SIM}_{toplevel}_d{os.environ.get('NQX_DIM', '128')}_i{iter_cordic}{tag}"
    build_dir = ROOT / "verif" / "sim_build" / name
    runner = get_runner(SIM)
    env = {"PYTHONPATH": os.pathsep.join([str(test_dir), str(MODEL), os.environ.get("PYTHONPATH", "")])}
    build_args = []
    if SIM == "icarus":
        build_args = ["-g2012"]
    elif SIM == "verilator":
        build_args = ["-Wno-fatal", "-Wno-WIDTH", "--x-assign", "unique", "--x-initial", "unique"]
    runner.build(
        sources=sources,
        hdl_toplevel=toplevel,
        includes=[RTL],
        build_dir=build_dir,
        build_args=build_args,
        parameters=parameters if parameters is not None else {"ITER_CORDIC": int(iter_cordic)},
        timescale=("1ns", "1ps"),
        always=True,
    )
    # The cocotb runner exports this process's sys.path as PYTHONPATH.
    sys.path[:0] = [str(test_dir), str(MODEL)]
    results = runner.test(
        hdl_toplevel=toplevel,
        test_module=module,
        test_dir=test_dir,
        build_dir=build_dir,
        extra_env={k: v for k, v in {**os.environ, **env}.items() if k.startswith(("NQX_", "COCOTB_", "PYTHONPATH"))},
    )
    total, failed = get_results(results)
    assert total > 0 and failed == 0, f"{failed} of {total} cocotb tests failed"


def _rtl(names: list[str]) -> list[Path]:
    return [RTL / n for n in names]


def test_cordic():
    _run("nqx_s1_cordic", _rtl(["nqx_s1_cordic.sv"]), "test_cordic", parameters={})


def test_core():
    _run("nqx_s1_core", _rtl(CORE_SOURCES), "test_core")


def test_top():
    _run("nqx_s1_top", _rtl(TOP_SOURCES), "test_top")


@pytest.mark.skipif(not IHP_PDK.exists(), reason="IHP PDK not found (set NQX_IHP_PDK)")
def test_chip_ihp():
    """Full IHP chip_top (pad cells + core RTL) through the pads."""
    io = IHP_PDK / "libs.ref" / "sg13g2_io" / "verilog" / "sg13g2_io.v"
    srcs = [IHP_FLOW / "sim" / "tb_chip.sv", IHP_FLOW / "chip_top.sv", io] + _rtl(TOP_SOURCES)
    _run("tb_chip", srcs, "test_top", "_ihp", parameters={})


@pytest.mark.skipif(
    not (IHP_PDK.exists() and IHP_NETLIST.exists()), reason="needs the IHP PDK and a netlist"
)
def test_chip_ihp_gate_level():
    """Post-route netlist of the IHP chip with the foundry cell models."""
    sys.path.insert(0, str(ROOT / "tools"))
    from functional_cells import convert

    ref = IHP_PDK / "libs.ref"
    cells = ROOT / "verif" / "sim_build" / "sg13g2_stdcell_functional.v"
    cells.parent.mkdir(parents=True, exist_ok=True)
    cells.write_text(convert((ref / "sg13g2_stdcell" / "verilog" / "sg13g2_stdcell.v").read_text()))
    srcs = [
        IHP_FLOW / "sim" / "tb_chip.sv",
        IHP_NETLIST,
        cells,
        ref / "sg13g2_io" / "verilog" / "sg13g2_io.v",
        IHP_FLOW / "sim" / "bondpad_70x70.v",
    ]
    _run("tb_chip", srcs, "test_top", "_ihp_gl", parameters={})


def test_tinytapeout():
    """The generated Tiny Tapeout package with its own testbench and test.

    Its Makefile is for Tiny Tapeout's CI; this runs the same files here.
    """
    tt = ROOT / "tinytapeout"
    srcs = [tt / "test" / "tb.v", *sorted((tt / "src").glob("*.v"))]
    _run("tb", srcs, "test", "_tt", test_dir=tt / "test", parameters={})
