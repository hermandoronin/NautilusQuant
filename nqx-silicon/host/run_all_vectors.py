"""Replay every golden-vector file against a chip (MicroPython) or the model (CPython).

MicroPython on the board:   import run_all_vectors; run_all_vectors.main()
CPython (self-check):       python host/run_all_vectors.py
"""

import sys

try:
    import os

    listdir = os.listdir
except ImportError:  # pragma: no cover
    import uos as os  # type: ignore

    listdir = os.listdir

import nqx_s1_host as h


def main(vector_dir="vectors", transport=None):
    files = sorted(f for f in listdir(vector_dir) if f.endswith(".txt") and f[0].isdigit())
    total_fail = 0
    for name in files:
        if transport is None:
            t = make_transport()
        else:
            t = transport
            t.reset()
        chip = h.NQXS1(t)
        passed, failed = h.run_vector_file(chip, vector_dir + "/" + name, verbose=True)
        total_fail += failed
        print("%-32s %4d passed %4d failed" % (name, passed, failed))
    print("RESULT:", "PASS" if total_fail == 0 else "FAIL")
    return total_fail == 0


def make_transport():
    if sys.implementation.name == "micropython":
        import nqx_pins

        t = nqx_pins.PinTransport()
        t.reset()
        return t
    return h.ModelTransport()


if __name__ == "__main__":
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "model"))
    sys.exit(0 if main(str(root / "vectors")) else 1)
