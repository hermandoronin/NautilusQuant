// Test wrapper: exposes chip_top's pads as the nqx_s1_top pin names so the
// pin-level cocotb tests (verif/cocotb/test_top.py) run unchanged on the
// wafer.space chip, at RTL or gate level.
`default_nettype none
`include "slot_defines.svh"

module tb_chip (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] in_bus,
    input  wire       in_req,
    output wire       in_ack,
    output wire [7:0] out_bus,
    output wire       out_req,
    input  wire       out_ack,
    output wire       busy
);
    wire [`NUM_BIDIR_PADS-1:0]  bidir;
    wire [`NUM_INPUT_PADS-1:0]  inputs;
    wire [`NUM_ANALOG_PADS-1:0] analog;

    assign bidir[7:0] = in_bus;
    assign inputs     = {{(`NUM_INPUT_PADS-2){1'b0}}, out_ack, in_req};
    assign out_bus    = bidir[15:8];
    assign in_ack     = bidir[16];
    assign out_req    = bidir[17];
    assign busy       = bidir[18];

`ifdef USE_POWER_PINS
    // Supply nets: chip_top's power ports are inout, so they cannot take
    // constants directly.
    supply1 vdd;
    supply0 vss;
`endif

    chip_top u_chip (
`ifdef USE_POWER_PINS
        .VDD(vdd), .VSS(vss),
`ifndef NQX_GL
        // RTL only: the powered netlist has VDD/VSS alone (VDD_NETS and
        // GND_NETS in librelane/config.yaml); DVDD/DVSS are tied to them.
        .DVDD(vdd), .DVSS(vss),
`endif
`endif
        .clk_PAD(clk), .rst_n_PAD(rst_n),
        .input_PAD(inputs), .bidir_PAD(bidir), .analog_PAD(analog)
    );
endmodule

`default_nettype wire
