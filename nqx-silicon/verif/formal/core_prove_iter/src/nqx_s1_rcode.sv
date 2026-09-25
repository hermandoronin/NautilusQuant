// NQX-S1 radius code: q = #{m in 1..7 : 14*d > (2m-1)*rng}, d = r - rmin,
// i.e. round(7*d/rng) with ties rounded down. Bit-exact with
// model/nqx_s1/quant.py:radius_code.
`default_nettype none
`include "nqx_s1_params.vh"

module nqx_s1_rcode (
    input  wire signed [`NQX_DW:0]   d,
    input  wire        [`NQX_DW-1:0] rng,
    output wire        [2:0]         q
);
    localparam integer W = `NQX_DW + 5;

    wire signed [W-1:0] dx  = {{4{d[`NQX_DW]}}, d};
    wire signed [W-1:0] d14 = (dx <<< 4) - (dx <<< 1);
    wire signed [W-1:0] r1  = $signed({5'b0, rng});
    wire signed [W-1:0] r2  = r1 <<< 1;
    wire signed [W-1:0] r4  = r1 <<< 2;
    wire signed [W-1:0] r8  = r1 <<< 3;
    wire signed [W-1:0] r16 = r1 <<< 4;

    wire signed [W-1:0] t1  = r1;
    wire signed [W-1:0] t3  = r2 + r1;
    wire signed [W-1:0] t5  = r4 + r1;
    wire signed [W-1:0] t7  = r8 - r1;
    wire signed [W-1:0] t9  = r8 + r1;
    wire signed [W-1:0] t11 = r8 + t3;
    wire signed [W-1:0] t13 = r16 - t3;

    wire [6:0] gt = {d14 > t13, d14 > t11, d14 > t9, d14 > t7, d14 > t5, d14 > t3, d14 > t1};

    assign q = {2'b0, gt[0]} + {2'b0, gt[1]} + {2'b0, gt[2]} + {2'b0, gt[3]}
             + {2'b0, gt[4]} + {2'b0, gt[5]} + {2'b0, gt[6]};
endmodule

`default_nettype wire
