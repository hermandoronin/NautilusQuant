# NQX-S1 placement-and-routing constraints (IHP SG13G2): the sign-off
# constraints in chip_top.sdc plus a tighter transition target for the
# resizer. Sign-off STA checks the library limit (max_transition 2.507 ns at
# the slow corner); repairing towards 1.5 ns leaves margin for the antenna
# diodes added after routing and makes weakly driven high-fanout nets
# faster at 1.08 V / 125 C.
source [file join [file dirname $::env(_SDC_IN)] chip_top.sdc]
set_max_transition 1.5 [current_design]
