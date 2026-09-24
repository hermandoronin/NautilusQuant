"""IHP SG13G2 metal/active fill in KLayout's hierarchical (deep) mode.

Drop-in replacement for the PDK's libs.tech/klayout/tech/scripts/filler.py,
with the same arguments:

    klayout -b -zz -r filler_deep.py -rd output_file=<out.gds> [-rd no_activ]
            [-rd no_metal] [-rd no_topmetal] [-rd threads=N] <in.gds>

It runs the PDK's own fill macros (sg13g2_filler_{ActGatP,Metal,TopMetal}.lym)
unchanged except for two engine directives prepended at run time: `deep` and
`threads(N)`. The stock script works on a flattened copy of the whole die;
on this 2 x 2 mm chip that needed more than 12 GB and was killed on a 16 GB
machine, while deep mode peaks at about 2 GB. The fill rules, patterns and
exclusions are the foundry's; the result is checked by the flow's density
and DRC steps like any other fill.
"""
# pylint: disable=import-error,undefined-variable

import os
import pathlib
import sys
import time

import pya

try:
    output_file
except NameError:
    print("Missing output_file argument. Please define '-rd output_file=<path-to-output-file>'")
    sys.exit(1)

try:
    n_threads = int(threads)
except NameError:
    n_threads = os.cpu_count() or 1

macros = pathlib.Path(os.environ["PDK_ROOT"]) / os.environ["PDK"] / "libs.tech/klayout/tech/macros"
for flag, area in (("no_activ", "ActGatP"), ("no_metal", "Metal"), ("no_topmetal", "TopMetal")):
    if flag in globals():
        print(f"Skip {area} fill because disabled by argument")
        continue
    print(f"Start filling {area} (deep mode, {n_threads} threads)", flush=True)
    t0 = time.time()
    macro = pya.Macro(str(macros / f"sg13g2_filler_{area}.lym"))
    macro.text = f"deep\nthreads({n_threads})\n" + macro.text
    macro.run()
    print(f"Done filling {area} in {time.time() - t0:.0f} s", flush=True)

pya.CellView.active().layout().write(output_file)
