/*
 * NQX-S1 (NautilusQuant golden-angle KV compression engine) for Tiny Tapeout.
 * SPDX-License-Identifier: Apache-2.0
 */
`default_nettype none

module tt_um_hermandoronin_nqx_s1 (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
    wire in_ack, out_req, busy;

    assign uio_oe  = 8'b0001_0110;
    assign uio_out = {3'b000, busy, 1'b0, out_req, in_ack, 1'b0};

    nqx_s1_top #(.ITER_CORDIC(1)) u_nqx (
        .clk     (clk),
        .rst_n   (rst_n),
        .in_bus  (ui_in),
        .in_req  (uio_in[0]),
        .in_ack  (in_ack),
        .out_bus (uo_out),
        .out_req (out_req),
        .out_ack (uio_in[3]),
        .busy    (busy)
    );

    wire _unused = &{ena, uio_in[7:4], uio_in[2:1], 1'b0};
endmodule

`default_nettype wire
