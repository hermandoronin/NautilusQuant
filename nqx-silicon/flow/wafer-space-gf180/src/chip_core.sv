// SPDX-License-Identifier: Apache-2.0
//
// NQX-S1 core for the wafer.space GF180MCU project template.
// chip_top.sv (template, unchanged) instantiates this module with the pad
// signals. Pad allocation:
//
//   input_in[0]        in_req   host -> chip request  (Schmitt, pull-down)
//   input_in[1]        out_ack  host -> chip ack      (Schmitt, pull-down)
//   bidir[7:0]         in_bus   host -> chip data     (input, Schmitt, pull-down)
//   bidir[15:8]        out_bus  chip -> host data     (output)
//   bidir[16]          in_ack   chip -> host ack      (output)
//   bidir[17]          out_req  chip -> host request  (output)
//   bidir[18]          busy     core busy flag        (output)
//   other pads         driven low
`default_nettype none

// The 0p5x0p5 slot builds with the iterative CORDIC (NQX_ITER_CORDIC=1, set in
// librelane/slots/slot_0p5x0p5.yaml): bit-identical results, 25 % less area.
`ifndef NQX_ITER_CORDIC
`define NQX_ITER_CORDIC 0
`endif

module chip_core #(
    parameter NUM_INPUT_PADS,
    parameter NUM_BIDIR_PADS,
    parameter NUM_ANALOG_PADS
    )(
    `ifdef USE_POWER_PINS
    inout  wire VDD,
    inout  wire VSS,
    `endif

    input  wire clk,
    input  wire rst_n,

    input  wire [NUM_INPUT_PADS-1:0] input_in,
    output wire [NUM_INPUT_PADS-1:0] input_pu,
    output wire [NUM_INPUT_PADS-1:0] input_pd,

    input  wire [NUM_BIDIR_PADS-1:0] bidir_in,
    output wire [NUM_BIDIR_PADS-1:0] bidir_out,
    output wire [NUM_BIDIR_PADS-1:0] bidir_oe,
    output wire [NUM_BIDIR_PADS-1:0] bidir_cs,
    output wire [NUM_BIDIR_PADS-1:0] bidir_sl,
    output wire [NUM_BIDIR_PADS-1:0] bidir_ie,
    output wire [NUM_BIDIR_PADS-1:0] bidir_pu,
    output wire [NUM_BIDIR_PADS-1:0] bidir_pd,

    inout  wire [NUM_ANALOG_PADS-1:0] analog
);
    localparam integer NB = NUM_BIDIR_PADS;

    // Bidirectional pads 0..7 are inputs, everything else is an output.
    localparam [NB-1:0] IN_MASK = {{(NB-8){1'b0}}, 8'hFF};

    wire [7:0] out_bus;
    wire       in_ack, out_req, busy;

    assign input_pu  = '0;
    assign input_pd  = '1;

    assign bidir_oe  = ~IN_MASK;
    assign bidir_ie  = IN_MASK;
    assign bidir_cs  = IN_MASK;
    assign bidir_pd  = IN_MASK;
    assign bidir_pu  = '0;
    assign bidir_sl  = '0;
    assign bidir_out = {{(NB-19){1'b0}}, busy, out_req, in_ack, out_bus, 8'h00};

    nqx_s1_top #(.ITER_CORDIC(`NQX_ITER_CORDIC)) u_nqx (
        .clk     (clk),
        .rst_n   (rst_n),
        .in_bus  (bidir_in[7:0]),
        .in_req  (input_in[0]),
        .in_ack  (in_ack),
        .out_bus (out_bus),
        .out_req (out_req),
        .out_ack (input_in[1]),
        .busy    (busy)
    );

    logic _unused;
    assign _unused = &{1'b0, input_in[NUM_INPUT_PADS-1:2], bidir_in[NB-1:8]};

endmodule

`default_nettype wire
