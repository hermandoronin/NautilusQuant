# NQX-S1 chip-level timing constraints (IHP SG13G2).
#
# One clock (clk_PAD -> clk_pad/p2c). The host interface is an asynchronous
# 4-phase handshake: in_req, out_ack and rst_n go through two-flop
# synchronizers, in_bus is sampled >= 2 cycles after in_req is seen, and
# out_bus changes one cycle before out_req rises. Those ports are therefore
# not timed against clk; set_max_delay keeps their pad-to-flop and
# flop-to-pad paths short and bounded instead.

current_design $::env(DESIGN_NAME)
set_units -time ns

set period $::env(CLOCK_PERIOD)
create_clock [get_pins clk_pad/p2c] -name clk -period $period
set_clock_uncertainty -setup $::env(CLOCK_UNCERTAINTY_CONSTRAINT) [get_clocks clk]
# Hold: 0.10 ns on top of the propagated clock, the +-5 % derate and the
# resizer's hold margin (0.10 ns after CTS, 0.05 ns after global routing),
# signed off at the fast corner. The setup value (0.25 ns) applied to hold
# made the resizer insert 7 254 hold buffers (+17.6 % cell area).
set_clock_uncertainty -hold 0.10 [get_clocks clk]
set_clock_transition $::env(CLOCK_TRANSITION_CONSTRAINT) [get_clocks clk]
set_propagated_clock [get_clocks clk]

set_timing_derate -early [expr {1 - $::env(TIME_DERATING_CONSTRAINT) / 100.0}]
set_timing_derate -late  [expr {1 + $::env(TIME_DERATING_CONSTRAINT) / 100.0}]

set_max_fanout $::env(MAX_FANOUT_CONSTRAINT) [current_design]
# Tighter than the library limit (2.507 ns at the slow corner): weakly driven
# high-fanout nets are both slow and the first to fail at 1.08 V/125 C.
set_max_transition 1.5 [current_design]

set async_in  [get_ports {rst_n_PAD in_req_PAD out_ack_PAD in_bus_PAD[*]}]
set async_out [get_ports {in_ack_PAD out_req_PAD out_bus_PAD[*] busy_PAD}]

set_max_delay [expr {0.6 * $period}] -from $async_in
set_max_delay [expr {0.6 * $period}] -to $async_out

set_load [expr {$::env(OUTPUT_CAP_LOAD) / 1000.0}] $async_out
