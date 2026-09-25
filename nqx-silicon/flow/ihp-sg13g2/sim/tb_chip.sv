// Test wrapper: exposes the IHP chip_top pads under the nqx_s1_top pin names
// so verif/cocotb/test_top.py runs unchanged on the chip (RTL + pad models,
// or the post-route gate-level netlist).
`default_nettype none

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
    chip_top u_chip (
`ifdef USE_POWER_PINS
        .VDD(1'b1), .VSS(1'b0), .IOVDD(1'b1), .IOVSS(1'b0),
`endif
        .clk_PAD(clk), .rst_n_PAD(rst_n),
        .in_bus_PAD(in_bus), .in_req_PAD(in_req), .in_ack_PAD(in_ack),
        .out_bus_PAD(out_bus), .out_req_PAD(out_req), .out_ack_PAD(out_ack),
        .busy_PAD(busy)
    );
endmodule

`default_nettype wire
