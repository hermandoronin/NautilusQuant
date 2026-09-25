// NQX-S1 radius residual sign: s = (7*d >= q*rng), the "QJL" bit of the
// radius. Bit-exact with model/nqx_s1/quant.py:radius_code.
`default_nettype none
`include "nqx_s1_params.vh"

module nqx_s1_rsign (
    input  wire signed [`NQX_DW:0]   d,
    input  wire        [`NQX_DW-1:0] rng,
    input  wire        [2:0]         q,
    output wire                      s
);
    localparam integer W = `NQX_DW + 4;

    wire signed [W-1:0] dx = {{3{d[`NQX_DW]}}, d};
    wire signed [W-1:0] d7 = (dx <<< 3) - dx;
    wire signed [W-1:0] r1 = $signed({4'b0, rng});
    wire signed [W-1:0] qr = (q[0] ? r1 : {W{1'b0}}) + (q[1] ? (r1 <<< 1) : {W{1'b0}})
                           + (q[2] ? (r1 <<< 2) : {W{1'b0}});
    assign s = d7 >= qr;
endmodule

`default_nettype wire
