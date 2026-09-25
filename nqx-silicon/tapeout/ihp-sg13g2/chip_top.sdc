###############################################################################
# Created by write_sdc
###############################################################################
current_design chip_top
###############################################################################
# Timing Constraints
###############################################################################
create_clock -name clk -period 20.0000 [get_pins {clk_pad/p2c}]
set_clock_transition 0.1500 [get_clocks {clk}]
set_clock_uncertainty -setup 0.2500 clk
set_clock_uncertainty -hold 0.1000 clk
set_propagated_clock [get_clocks {clk}]
set_max_delay\
    -from [list [get_ports {in_bus_PAD[0]}]\
           [get_ports {in_bus_PAD[1]}]\
           [get_ports {in_bus_PAD[2]}]\
           [get_ports {in_bus_PAD[3]}]\
           [get_ports {in_bus_PAD[4]}]\
           [get_ports {in_bus_PAD[5]}]\
           [get_ports {in_bus_PAD[6]}]\
           [get_ports {in_bus_PAD[7]}]\
           [get_ports {in_req_PAD}]\
           [get_ports {out_ack_PAD}]\
           [get_ports {rst_n_PAD}]] 12.0000
set_max_delay\
    -to [list [get_ports {busy_PAD}]\
           [get_ports {in_ack_PAD}]\
           [get_ports {out_bus_PAD[0]}]\
           [get_ports {out_bus_PAD[1]}]\
           [get_ports {out_bus_PAD[2]}]\
           [get_ports {out_bus_PAD[3]}]\
           [get_ports {out_bus_PAD[4]}]\
           [get_ports {out_bus_PAD[5]}]\
           [get_ports {out_bus_PAD[6]}]\
           [get_ports {out_bus_PAD[7]}]\
           [get_ports {out_req_PAD}]] 12.0000
###############################################################################
# Environment
###############################################################################
set_load -pin_load 0.0060 [get_ports {busy_PAD}]
set_load -pin_load 0.0060 [get_ports {in_ack_PAD}]
set_load -pin_load 0.0060 [get_ports {out_req_PAD}]
set_load -pin_load 0.0060 [get_ports {out_bus_PAD[7]}]
set_load -pin_load 0.0060 [get_ports {out_bus_PAD[6]}]
set_load -pin_load 0.0060 [get_ports {out_bus_PAD[5]}]
set_load -pin_load 0.0060 [get_ports {out_bus_PAD[4]}]
set_load -pin_load 0.0060 [get_ports {out_bus_PAD[3]}]
set_load -pin_load 0.0060 [get_ports {out_bus_PAD[2]}]
set_load -pin_load 0.0060 [get_ports {out_bus_PAD[1]}]
set_load -pin_load 0.0060 [get_ports {out_bus_PAD[0]}]
set_timing_derate -early 0.9500
set_timing_derate -late 1.0500
###############################################################################
# Design Rules
###############################################################################
set_max_transition 1.5000 [current_design]
set_max_fanout 10.0000 [current_design]
