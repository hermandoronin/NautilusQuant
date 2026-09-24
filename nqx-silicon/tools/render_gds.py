# Render a GDS layout to PNG with KLayout (batch mode).
#
#   klayout -b -r tools/render_gds.py -rd gds=chip_top.gds -rd lyp=<layer props> \
#           -rd out=chip.png [-rd width=2400]
#
# Uses the PDK's layer properties so the picture looks like the foundry view.
import pya

width = int(globals().get("width", "2400"))
view = pya.LayoutView()
view.load_layout(gds, 0)  # noqa: F821  (set with -rd)
if globals().get("lyp"):
    view.load_layer_props(lyp)  # noqa: F821
view.max_hier()
view.zoom_fit()
view.set_config("background-color", "#ffffff")
view.set_config("grid-visible", "false")
view.set_config("text-visible", "false")
bbox = view.active_cellview().cell.dbbox()
height = int(width * bbox.height() / bbox.width())
view.save_image(out, width, height)  # noqa: F821
print(f"wrote {out} ({width}x{height})")  # noqa: F821
