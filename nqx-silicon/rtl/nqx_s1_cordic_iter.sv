// NQX-S1 iterative CORDIC: the same arithmetic as nqx_s1_cordic.sv, one
// micro-rotation per clock on a single set of registers. Bit-exact with the
// pipelined version and with model/nqx_s1/cordic.py; accepts a new operand
// pair every NITER+1 cycles (in_ready) instead of every cycle. Used for
// area-constrained builds (Tiny Tapeout).
`default_nettype none
`include "nqx_s1_params.vh"

module nqx_s1_cordic_iter #(
    parameter integer TAG_W = 14
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        in_valid,
    output wire                        in_ready,
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
    output reg         [TAG_W-1:0]     out_tag,
    output wire                        busy
);
    localparam integer DW = `NQX_DW;
    localparam integer CW = `NQX_CW;
    localparam integer ZW = `NQX_ZW;
    localparam integer GB = `NQX_GB;
    localparam integer N  = `NQX_NITER;
    localparam integer KF = `NQX_KF;
    localparam integer SH = KF + GB;
    localparam integer PW = CW + KF + 1;
    localparam integer IW = $clog2(N + 1);
    localparam [IW-1:0] LAST = IW'(N - 1);

    `include "nqx_s1_atan.vh"

    // Quadrant pre-rotation: identical to the pipelined version.
    wire signed [CW-1:0] ext_x = {{(CW-DW-GB){in_x[DW-1]}}, in_x, {GB{1'b0}}};
    wire signed [CW-1:0] ext_y = {{(CW-DW-GB){in_y[DW-1]}}, in_y, {GB{1'b0}}};
    wire rot_flip = in_z[ZW-1] ^ in_z[ZW-2];
    wire vec_flip = in_x[DW-1];
    wire flip     = in_vec ? vec_flip : rot_flip;
    wire [ZW-1:0] z_rot = {in_z[ZW-1] ^ rot_flip, in_z[ZW-2:0]};
    wire [ZW-1:0] z_vec = {vec_flip, {(ZW-1){1'b0}}};

    reg                 loaded;     // operands captured, iterations pending
    reg                 fin;        // iterations done, output stage next
    reg  [IW-1:0]       i;
    reg  signed [CW-1:0] x, y;
    reg         [ZW-1:0] z;
    reg                 m;
    reg  [TAG_W-1:0]    t;

    assign in_ready = !loaded && !fin;
    assign busy     = loaded || fin || out_valid;

    wire signed [CW-1:0] xsh = x >>> i;
    wire signed [CW-1:0] ysh = y >>> i;
    wire                 d   = m ? y[CW-1] : ~z[ZW-1];
    wire        [ZW-1:0] a   = nqx_atan({{(32-IW){1'b0}}, i});

    // Gain compensation and saturation: identical to the pipelined version.
    wire signed [KF:0]   kinv = `NQX_KINV;
    wire signed [PW-1:0] rnd  = 1 <<< (SH-1);
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [PW-1:0]    px = x * kinv + rnd;
    wire signed [PW-1:0]    py = y * kinv + rnd;
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
            loaded    <= 1'b0;
            fin       <= 1'b0;
            i         <= {IW{1'b0}};
            x         <= {CW{1'b0}};
            y         <= {CW{1'b0}};
            z         <= {ZW{1'b0}};
            m         <= 1'b0;
            t         <= {TAG_W{1'b0}};
            out_valid <= 1'b0;
            out_x     <= {DW{1'b0}};
            out_y     <= {DW{1'b0}};
            out_z     <= {ZW{1'b0}};
            out_sat   <= 1'b0;
            out_tag   <= {TAG_W{1'b0}};
        end else begin
            out_valid <= 1'b0;
            if (in_valid && in_ready) begin
                x      <= flip ? -ext_x : ext_x;
                y      <= flip ? -ext_y : ext_y;
                z      <= in_vec ? z_vec : z_rot;
                m      <= in_vec;
                t      <= in_tag;
                i      <= {IW{1'b0}};
                loaded <= 1'b1;
            end else if (loaded) begin
                x <= d ? x - ysh : x + ysh;
                y <= d ? y + xsh : y - xsh;
                z <= d ? z - a : z + a;
                i <= i + 1'b1;
                if (i == LAST) begin
                    loaded <= 1'b0;
                    fin    <= 1'b1;
                end
            end else if (fin) begin
                fin       <= 1'b0;
                out_valid <= 1'b1;
                out_x     <= sx_hi ? HI[DW-1:0] : sx_lo ? LO[DW-1:0] : cx[DW-1:0];
                out_y     <= sy_hi ? HI[DW-1:0] : sy_lo ? LO[DW-1:0] : cy[DW-1:0];
                out_z     <= z;
                out_sat   <= sx_hi | sx_lo | (~m & (sy_hi | sy_lo));
                out_tag   <= t;
            end
        end
    end
endmodule

`default_nettype wire
