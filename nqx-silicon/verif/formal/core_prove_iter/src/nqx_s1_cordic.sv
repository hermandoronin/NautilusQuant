// NQX-S1 pipelined CORDIC.
//
// Mode in_vec = 0 (rotation):  (x, y) rotated by angle z.
// Mode in_vec = 1 (vectoring): (x, y) -> (r, theta); r on out_x, theta on out_z.
//
// Angles are ZW-bit binary angles (full turn = 2^ZW, so mod-2*pi is free).
// Latency is NITER + 2 cycles, one new operand pair per cycle.
// Bit-exact with model/nqx_s1/cordic.py.
`default_nettype none
`include "nqx_s1_params.vh"

module nqx_s1_cordic #(
    parameter integer TAG_W = 14
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        in_valid,
    input  wire                        in_vec,
    input  wire signed [`NQX_DW-1:0]   in_x,
    input  wire signed [`NQX_DW-1:0]   in_y,
    input  wire        [`NQX_ZW-1:0]   in_z,
    input  wire        [TAG_W-1:0]     in_tag,
    output reg                         out_valid,
    output reg  signed [`NQX_DW-1:0]   out_x,
    output reg  signed [`NQX_DW-1:0]   out_y,
    output reg         [`NQX_ZW-1:0]   out_z,
    output reg                         out_sat,
    output reg         [TAG_W-1:0]     out_tag
);

    localparam integer DW = `NQX_DW;
    localparam integer CW = `NQX_CW;
    localparam integer ZW = `NQX_ZW;
    localparam integer GB = `NQX_GB;
    localparam integer N  = `NQX_NITER;
    localparam integer KF = `NQX_KF;
    localparam integer SH = KF + GB;
    localparam integer PW = CW + KF + 1;

    `include "nqx_s1_atan.vh"

    // Stage outputs, stage s drives slice s.
    wire [CW*(N+1)-1:0]    xs;
    wire [CW*(N+1)-1:0]    ys;
    wire [ZW*(N+1)-1:0]    zs;
    wire [N:0]             vs;
    wire [N:0]             ms;
    wire [TAG_W*(N+1)-1:0] ts;

    // ---------------------------------------------------------- stage 0 ----
    // Quadrant pre-rotation so that the residual angle lies in [-90, 90).
    wire signed [CW-1:0] ext_x = {{(CW-DW-GB){in_x[DW-1]}}, in_x, {GB{1'b0}}};
    wire signed [CW-1:0] ext_y = {{(CW-DW-GB){in_y[DW-1]}}, in_y, {GB{1'b0}}};

    wire rot_flip = in_z[ZW-1] ^ in_z[ZW-2];
    wire vec_flip = in_x[DW-1];
    wire flip     = in_vec ? vec_flip : rot_flip;

    wire [ZW-1:0] z_rot = {in_z[ZW-1] ^ rot_flip, in_z[ZW-2:0]};
    wire [ZW-1:0] z_vec = {vec_flip, {(ZW-1){1'b0}}};

    reg              v0_q;
    reg [CW-1:0]     x0_q;
    reg [CW-1:0]     y0_q;
    reg [ZW-1:0]     z0_q;
    reg              m0_q;
    reg [TAG_W-1:0]  t0_q;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v0_q <= 1'b0;
            x0_q <= {CW{1'b0}};
            y0_q <= {CW{1'b0}};
            z0_q <= {ZW{1'b0}};
            m0_q <= 1'b0;
            t0_q <= {TAG_W{1'b0}};
        end else begin
            v0_q <= in_valid;
            x0_q <= flip ? -ext_x : ext_x;
            y0_q <= flip ? -ext_y : ext_y;
            z0_q <= in_vec ? z_vec : z_rot;
            m0_q <= in_vec;
            t0_q <= in_tag;
        end
    end

    assign vs[0]            = v0_q;
    assign xs[CW-1:0]       = x0_q;
    assign ys[CW-1:0]       = y0_q;
    assign zs[ZW-1:0]       = z0_q;
    assign ms[0]            = m0_q;
    assign ts[TAG_W-1:0]    = t0_q;

    // ------------------------------------------------------- iterations ----
    genvar s;
    generate
        for (s = 0; s < N; s = s + 1) begin : g_iter
            wire signed [CW-1:0] x = xs[s*CW +: CW];
            wire signed [CW-1:0] y = ys[s*CW +: CW];
            wire        [ZW-1:0] z = zs[s*ZW +: ZW];
            wire signed [CW-1:0] xsh = x >>> s;
            wire signed [CW-1:0] ysh = y >>> s;
            wire          d = ms[s] ? y[CW-1] : ~z[ZW-1];
            wire [ZW-1:0] a = nqx_atan(s);

            reg             v_q;
            reg [CW-1:0]    x_q;
            reg [CW-1:0]    y_q;
            reg [ZW-1:0]    z_q;
            reg             m_q;
            reg [TAG_W-1:0] t_q;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    v_q <= 1'b0;
                    x_q <= {CW{1'b0}};
                    y_q <= {CW{1'b0}};
                    z_q <= {ZW{1'b0}};
                    m_q <= 1'b0;
                    t_q <= {TAG_W{1'b0}};
                end else begin
                    v_q <= vs[s];
                    x_q <= d ? x - ysh : x + ysh;
                    y_q <= d ? y + xsh : y - xsh;
                    z_q <= d ? z - a : z + a;
                    m_q <= ms[s];
                    t_q <= ts[s*TAG_W +: TAG_W];
                end
            end

            assign vs[s+1]                  = v_q;
            assign xs[(s+1)*CW +: CW]       = x_q;
            assign ys[(s+1)*CW +: CW]       = y_q;
            assign zs[(s+1)*ZW +: ZW]       = z_q;
            assign ms[s+1]                  = m_q;
            assign ts[(s+1)*TAG_W +: TAG_W] = t_q;
        end
    endgenerate

    // ---------------------------------------------- gain compensation ----
    wire signed [CW-1:0] xn   = xs[N*CW +: CW];
    wire signed [CW-1:0] yn   = ys[N*CW +: CW];
    wire signed [KF:0]   kinv = `NQX_KINV;
    wire signed [PW-1:0] rnd  = 1 <<< (SH-1);

    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [PW-1:0]    px = xn * kinv + rnd;   // low SH bits are rounded off
    wire signed [PW-1:0]    py = yn * kinv + rnd;
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [PW-SH-1:0] cx = px[PW-1:SH];
    wire signed [PW-SH-1:0] cy = py[PW-1:SH];

    localparam signed [PW-SH-1:0] HI = (1 <<< (DW-1)) - 1;
    localparam signed [PW-SH-1:0] LO = -(1 <<< (DW-1));

    wire sx_hi = cx > HI;
    wire sx_lo = cx < LO;
    wire sy_hi = cy > HI;
    wire sy_lo = cy < LO;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_x     <= {DW{1'b0}};
            out_y     <= {DW{1'b0}};
            out_z     <= {ZW{1'b0}};
            out_sat   <= 1'b0;
            out_tag   <= {TAG_W{1'b0}};
        end else begin
            out_valid <= vs[N];
            out_x     <= sx_hi ? HI[DW-1:0] : sx_lo ? LO[DW-1:0] : cx[DW-1:0];
            out_y     <= sy_hi ? HI[DW-1:0] : sy_lo ? LO[DW-1:0] : cy[DW-1:0];
            out_z     <= zs[N*ZW +: ZW];
            out_sat   <= sx_hi | sx_lo | (~ms[N] & (sy_hi | sy_lo));
            out_tag   <= ts[N*TAG_W +: TAG_W];
        end
    end

endmodule

`default_nettype wire
