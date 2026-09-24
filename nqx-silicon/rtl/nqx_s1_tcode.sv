// NQX-S1 angle code: circular 8-sector quantizer of a ZW-bit binary angle
// plus residual sign. Bit-exact with model/nqx_s1/quant.py:angle_code.
`default_nettype none
`include "nqx_s1_params.vh"

module nqx_s1_tcode (
    input  wire [`NQX_ZW-1:0] t,
    output wire [2:0]         q,
    output wire               s
);
    localparam integer ZW = `NQX_ZW;

    wire [2:0]    tq = t[ZW-1:ZW-3] + {2'b0, t[ZW-4]};
    wire [ZW-1:0] e  = t - {tq, {(ZW-3){1'b0}}};

    assign q = tq;
    assign s = ~e[ZW-1];
endmodule

`default_nettype wire
