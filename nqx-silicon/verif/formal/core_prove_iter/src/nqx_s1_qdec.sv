// NQX-S1 dequantizer: packed byte -> (r_hat, theta_hat).
// Bit-exact with model/nqx_s1/quant.py:decode_pair.
`default_nettype none
`include "nqx_s1_params.vh"

module nqx_s1_qdec (
    input  wire        [7:0]                         pk,
    input  wire        [`NQX_DW-1:0]                 rmin,
    input  wire        [`NQX_DW+`NQX_STEPF-5:0]      step,
    input  wire                                      refine,
    output wire signed [`NQX_DW-1:0]                 r,
    output wire        [`NQX_ZW-1:0]                 t,
    output wire                                      sat
);
    localparam integer DW = `NQX_DW;
    localparam integer ZW = `NQX_ZW;
    localparam integer F  = `NQX_STEPF;
    localparam integer SW = DW + F - 4;
    localparam integer PW = SW + 6;

    wire [2:0] qr = pk[2:0];
    wire       sr = pk[3];
    wire [2:0] qt = pk[6:4];
    wire       st = pk[7];

    wire signed [5:0] adj = refine ? (sr ? 6'sd1 : -6'sd1) : 6'sd0;
    wire signed [5:0] num = $signed({1'b0, qr, 2'b00}) + adj;

    wire signed [PW-1:0] rnd  = 1 <<< (F - 1);
    wire signed [PW-1:0] prod = num * $signed({1'b0, step}) + rnd;
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [PW-1:0] prod_q = prod;
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [PW-F-1:0] term = prod_q[PW-1:F];
    wire signed [PW-F:0]   rsum = $signed({2'b0, rmin}) + $signed({term[PW-F-1], term});

    localparam signed [PW-F:0] RMAX = (1 <<< (DW-1)) - 1;
    assign sat = rsum > RMAX;
    assign r   = sat ? RMAX[DW-1:0] : rsum[DW-1:0];

    localparam [ZW-1:0] QUARTER = 1 << (ZW - 5);

    wire [ZW-1:0] tbase = {qt, {(ZW-3){1'b0}}};
    assign t = !refine ? tbase : st ? tbase + QUARTER : tbase - QUARTER;
endmodule

`default_nettype wire
