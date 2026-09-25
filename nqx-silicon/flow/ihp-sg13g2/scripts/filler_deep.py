"""IHP SG13G2 metal/active fill in KLayout's hierarchical (deep) mode.

Drop-in replacement for the PDK's libs.tech/klayout/tech/scripts/filler.py,
with the same arguments:

    klayout -b -zz -r filler_deep.py -rd output_file=<out.gds> [-rd no_activ]
            [-rd no_metal] [-rd no_topmetal] [-rd threads=N]
            [-rd metal_fill_distance=D] <in.gds>

It runs the PDK's own fill macros (sg13g2_filler_{ActGatP,Metal,TopMetal}.lym)
unchanged except for two engine directives prepended at run time: `deep` and
`threads(N)`. The stock script works on a flattened copy of the whole die;
on this 2 x 2 mm chip that needed more than 12 GB and was killed on a 16 GB
machine, while deep mode peaks at about 2 GB. The fill rules, patterns and
exclusions are the foundry's; the result is checked by the flow's density
and DRC steps like any other fill.

metal_fill_distance sets the Metal2..Metal5 filler-to-filler distance, the
parameter the PDK macro exposes in its interactive dialog (defaults 1.5 um
for Metal2 and 2.0 um for Metal3-5; the macro and rule MxFil.b allow down
to 0.42 um). Give one value for all four layers or four comma-separated
values (Metal2,Metal3,Metal4,Metal5). Batch mode has no dialog, so the
values are substituted into the macro's defaults. Metal1 keeps the PDK
default. IHP's filler documentation describes adjusting these parameters to
reach the density targets.
"""
# pylint: disable=import-error,undefined-variable

import os
import pathlib
import re
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

try:
    _d = [max(0.42, float(v)) for v in str(metal_fill_distance).split(",")]
    fill_distance = dict(zip((2, 3, 4, 5), _d * 4 if len(_d) == 1 else _d))
    if len(_d) not in (1, 4):
        print("metal_fill_distance takes 1 or 4 comma-separated values")
        sys.exit(1)
except NameError:
    fill_distance = None

macros = pathlib.Path(os.environ["PDK_ROOT"]) / os.environ["PDK"] / "libs.tech/klayout/tech/macros"
for flag, area in (("no_activ", "ActGatP"), ("no_metal", "Metal"), ("no_topmetal", "TopMetal")):
    if flag in globals():
        print(f"Skip {area} fill because disabled by argument")
        continue
    print(f"Start filling {area} (deep mode, {n_threads} threads)", flush=True)
    t0 = time.time()
    macro = pya.Macro(str(macros / f"sg13g2_filler_{area}.lym"))
    text = macro.text
    if area == "Metal" and fill_distance is not None:
        for layer, dist in fill_distance.items():
            text, n = re.subn(rf"('distance_m{layer}'\s*=>\s*)[0-9.]+", rf"\g<1>{dist}", text)
            if n != 1:
                print(f"Expected one Metal{layer} distance default in the PDK macro, found {n}")
                sys.exit(1)
        print("Metal2-5 filler distance [um]: " + ", ".join(f"M{k} {v}" for k, v in fill_distance.items()))
    macro.text = f"deep\nthreads({n_threads})\n" + text
    macro.run()
    print(f"Done filling {area} in {time.time() - t0:.0f} s", flush=True)

pya.CellView.active().layout().write(output_file)
