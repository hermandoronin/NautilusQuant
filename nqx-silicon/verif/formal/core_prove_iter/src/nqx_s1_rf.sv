// NQX-S1 vector register: DIM x DW flip-flops (cleared by reset), two
// asynchronous read ports and two write ports. Givens pairs within one layer
// are disjoint, so the two write ports never target the same element in
// normal operation; port B wins if they do.
`default_nettype none
`include "nqx_s1_params.vh"

module nqx_s1_rf (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire [`NQX_LOG2_DIM-1:0]    ra,
    input  wire [`NQX_LOG2_DIM-1:0]    rb,
    output wire signed [`NQX_DW-1:0]   rda,
    output wire signed [`NQX_DW-1:0]   rdb,
    input  wire                        wea,
    input  wire [`NQX_LOG2_DIM-1:0]    wa,
    input  wire signed [`NQX_DW-1:0]   wda,
    input  wire                        web,
    input  wire [`NQX_LOG2_DIM-1:0]    wb,
    input  wire signed [`NQX_DW-1:0]   wdb
);
    localparam integer DIM = `NQX_DIM;
    localparam integer DW  = `NQX_DW;

    wire [DW*DIM-1:0] flat;

    genvar e;
    generate
        for (e = 0; e < DIM; e = e + 1) begin : g_el
            reg [DW-1:0] q;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    q <= {DW{1'b0}};
                else if (web && wb == e)
                    q <= wdb;
                else if (wea && wa == e)
                    q <= wda;
            end
            assign flat[e*DW +: DW] = q;
        end
    endgenerate

    assign rda = flat[ra*DW +: DW];
    assign rdb = flat[rb*DW +: DW];
endmodule

`default_nettype wire
