// NQX-S1 serial divider: step = floor(rng * 2^STEPF / 28), one quotient bit
// per cycle, DW+STEPF cycles. Bit-exact with model/nqx_s1/quant.py:radius_step.
`default_nettype none
`include "nqx_s1_params.vh"

module nqx_s1_div28 (
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           start,
    input  wire [`NQX_DW-1:0]             rng,
    output reg  [`NQX_DW+`NQX_STEPF-5:0]  step,
    output reg                            busy
);
    localparam integer DW    = `NQX_DW;
    localparam integer F     = `NQX_STEPF;
    localparam integer NW    = DW + F;
    localparam integer CNT_W = $clog2(NW + 1);
    localparam [CNT_W-1:0] NBITS = CNT_W'(NW);

    reg [NW-1:0]    num;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [NW-1:0]    quo;       // top 4 bits are always zero: rng < 2^(DW-1)
    /* verilator lint_on UNUSEDSIGNAL */
    reg [4:0]       rem;
    reg [CNT_W-1:0] cnt;

    wire [5:0] rs   = {rem, num[NW-1]};
    wire       ge   = rs >= 6'd28;
    wire [4:0] rsub = rs[4:0] - 5'd28;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            num  <= {NW{1'b0}};
            quo  <= {NW{1'b0}};
            rem  <= 5'd0;
            cnt  <= {CNT_W{1'b0}};
            step <= {(NW-4){1'b0}};
        end else if (start) begin
            busy <= 1'b1;
            num  <= {rng, {F{1'b0}}};
            quo  <= {NW{1'b0}};
            rem  <= 5'd0;
            cnt  <= NBITS;
        end else if (busy) begin
            num <= num << 1;
            quo <= {quo[NW-2:0], ge};
            rem <= ge ? rsub : rs[4:0];
            cnt <= cnt - 1'b1;
            if (cnt == 1) begin
                busy <= 1'b0;
                step <= {quo[NW-6:0], ge};
            end
        end
    end
endmodule

`default_nettype wire
