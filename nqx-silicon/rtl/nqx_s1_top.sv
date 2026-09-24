// NQX-S1 top level (pad-facing ports of the digital block).
//
// 22 signal pins: clk, rst_n, in_bus[7:0], in_req, in_ack, out_bus[7:0],
// out_req, out_ack, busy.
`default_nettype none

module nqx_s1_top #(
    parameter integer ITER_CORDIC = 0
) (
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
    // Asynchronous assert, synchronous de-assert.
    /* verilator lint_off SYNCASYNCNET */
    reg [1:0] rst_sync;
    /* verilator lint_on SYNCASYNCNET */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_sync <= 2'b00;
        else        rst_sync <= {rst_sync[0], 1'b1};
    end
    wire rst_n_s = rst_sync[1];

    wire [7:0] s_data;
    wire       s_valid;
    wire       s_ready;
    wire [7:0] m_data;
    wire       m_valid;
    wire       m_ready;

    nqx_s1_hsio u_io (
        .clk(clk), .rst_n(rst_n_s),
        .in_bus(in_bus), .in_req(in_req), .in_ack(in_ack),
        .out_bus(out_bus), .out_req(out_req), .out_ack(out_ack),
        .s_data(s_data), .s_valid(s_valid), .s_ready(s_ready),
        .m_data(m_data), .m_valid(m_valid), .m_ready(m_ready)
    );

    nqx_s1_core #(.ITER_CORDIC(ITER_CORDIC)) u_core (
        .clk(clk), .rst_n(rst_n_s),
        .in_data(s_data), .in_valid(s_valid), .in_ready(s_ready),
        .out_data(m_data), .out_valid(m_valid), .out_ready(m_ready),
        .busy(busy)
    );
endmodule

`default_nettype wire
