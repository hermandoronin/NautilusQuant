// NQX-S1 core: command decoder, micro-op sequencer, CSRs, vector register,
// CORDIC, quantizer and dequantizer. Byte-stream interface with valid/ready
// handshake on both sides. Behaviour is defined by model/nqx_s1/core.py.
`default_nettype none
`include "nqx_s1_params.vh"

module nqx_s1_core #(
    // 0: pipelined CORDIC, one pair per clock (default).
    // 1: iterative CORDIC, one pair per NITER+1 clocks, much smaller.
    // Results are bit-identical; only the cycle counts differ.
    parameter integer ITER_CORDIC = 0
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] in_data,
    input  wire       in_valid,
    output wire       in_ready,
    output wire [7:0] out_data,
    output wire       out_valid,
    input  wire       out_ready,
    output wire       busy
);
    localparam integer DIM  = `NQX_DIM;
    localparam integer L    = `NQX_LOG2_DIM;
    localparam integer DW   = `NQX_DW;
    localparam integer ZW   = `NQX_ZW;
    localparam integer FW   = `NQX_FW;
    localparam integer HB   = `NQX_HB;
    localparam integer SW   = `NQX_DW + `NQX_STEPF - 4;
    localparam integer S3   = `NQX_L3_STRIDE;
    localparam integer S3B  = $clog2(S3);
    localparam integer NP   = `NQX_NPAIRS;
    localparam integer HBW  = 8 * HB;

    localparam [L-1:0] S3V      = L'(S3);
    localparam [L-1:0] KMAX_ADJ = L'(NP - 1);
    localparam [L-1:0] KMAX_SH  = L'(NP - 2);
    localparam [L-1:0] KMAX_L3  = L'(DIM - 1);
    localparam [L-1:0] KLAST_EL = L'(DIM - 1);
    localparam [2:0]   HB_LAST  = 3'(HB - 1);
    localparam [2:0]   HDR_LAST = 3'(2 * HB - 1);

    // ------------------------------------------------------------ opcodes
    localparam [7:0] OP_NOP = 8'h00, OP_LDV = 8'h01, OP_STV = 8'h02, OP_STVR = 8'h03,
                     OP_GVNS = 8'h10, OP_GVNS_INV = 8'h11, OP_POLAR = 8'h20,
                     OP_IPOLAR = 8'h21, OP_QUANT = 8'h30, OP_DEQUANT = 8'h31,
                     OP_ENC = 8'h60, OP_DEC = 8'h61, OP_SYNC = 8'h70, OP_CSRW = 8'h71,
                     OP_CSRR = 8'h72;

    localparam [7:0] SYNC_BYTE = 8'hA5;

    // ---------------------------------------------------------- micro-ops
    localparam [2:0] U_LDV = 3'd0, U_STV = 3'd1, U_STVR = 3'd2, U_ROT = 3'd3,
                     U_POLAR = 3'd4, U_IPOLAR = 3'd5, U_QUANT = 3'd6, U_DEQUANT = 3'd7;

    localparam [1:0] M_NONE = 2'd0, M_ENC = 2'd1, M_DEC = 2'd2;

    // ------------------------------------------------------------- states
    localparam [4:0] S_IDLE = 5'd0, S_ARG = 5'd1, S_CSRW_A = 5'd2, S_CSRW_D = 5'd3,
                     S_CSRR_A = 5'd4, S_EMIT = 5'd5, S_LAUNCH = 5'd6, S_NEXT = 5'd7,
                     S_LDV = 5'd8, S_STV = 5'd9, S_ISSUE = 5'd10, S_DRAIN = 5'd11,
                     S_QHDR = 5'd12, S_QRD = 5'd13, S_QC1 = 5'd14, S_QC2 = 5'd15,
                     S_DHDR = 5'd16, S_DPREP = 5'd17, S_DDIV = 5'd18, S_DBYTE = 5'd19,
                     S_DWR = 5'd20, S_CSRR_L = 5'd21;

    // --------------------------------------------------------- CSR space
    localparam [7:0] CSR_ID = 8'h00, CSR_VERSION = 8'h01, CSR_PARAMS = 8'h02,
                     CSR_CTRL = 8'h03, CSR_STATUS = 8'h04, CSR_INC0 = 8'h05,
                     CSR_INC1 = 8'h06, CSR_INC2 = 8'h07, CSR_RMIN = 8'h08,
                     CSR_RMAX = 8'h09, CSR_SCRATCH = 8'h0A, CSR_FEATURES = 8'h0B,
                     CSR_CYCLES = 8'h10,
                     CSR_BUSY = 8'h11, CSR_CORDIC = 8'h12, CSR_ENC = 8'h13,
                     CSR_DEC = 8'h14;

    // Macro sequences: ENC = LDV, ROT0, ROT1, ROT2, POLAR, QUANT
    //                  DEC = DEQUANT, IPOLAR, ROT2^-1, ROT1^-1, ROT0^-1, STV
    function automatic [5:0] macro_uop(input [1:0] m, input [2:0] step);
        // returns {uop[2:0], layer[1:0], inv}
        begin
            if (m == M_ENC) begin
                case (step)
                    3'd0:    macro_uop = {U_LDV, 2'd0, 1'b0};
                    3'd1:    macro_uop = {U_ROT, 2'd0, 1'b0};
                    3'd2:    macro_uop = {U_ROT, 2'd1, 1'b0};
                    3'd3:    macro_uop = {U_ROT, 2'd2, 1'b0};
                    3'd4:    macro_uop = {U_POLAR, 2'd0, 1'b0};
                    default: macro_uop = {U_QUANT, 2'd0, 1'b0};
                endcase
            end else begin
                case (step)
                    3'd0:    macro_uop = {U_DEQUANT, 2'd0, 1'b0};
                    3'd1:    macro_uop = {U_IPOLAR, 2'd0, 1'b0};
                    3'd2:    macro_uop = {U_ROT, 2'd2, 1'b1};
                    3'd3:    macro_uop = {U_ROT, 2'd1, 1'b1};
                    3'd4:    macro_uop = {U_ROT, 2'd0, 1'b1};
                    default: macro_uop = {U_STV, 2'd0, 1'b0};
                endcase
            end
        end
    endfunction

    // --------------------------------------------------------- registers
    reg [4:0]  state;
    reg [2:0]  uop;
    reg [1:0]  layer;
    reg        inv;
    reg [1:0]  macro;
    reg [2:0]  mstep;

    reg [L-1:0] k;       // pair / element counter
    reg [2:0]   b;       // byte counter within an element or header
    reg [7:0]   lo;      // low byte of an LDV element
    reg [31:0]  acc;     // phase accumulator
    reg         first;   // first radius of a POLAR pass

    reg [7:0]   caddr;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [31:0]  wshift;     // [7:0] is shifted out before the word completes
    /* verilator lint_on UNUSEDSIGNAL */
    reg [31:0]  oshift;
    reg [2:0]   ocnt;

    reg [7:0]   obuf;
    reg         obuf_v;

    reg [31:0]  inc0, inc1, inc2;
    reg         refine;
    reg         st_illegal, st_sat;
    reg [DW-1:0] rmin, rmax;
    reg [31:0]  scratch;
    reg [31:0]  cnt_cycles, cnt_busy, cnt_cordic, cnt_enc, cnt_dec;

    reg signed [DW:0] qd;
    reg [ZW-1:0]      qt;
    reg [2:0]         qq;
    reg [DW-1:0]      qrng;

    reg [8*2*HB-1:0]  hdr;
    wire [31:0]       wword = {in_data, wshift[31:8]};
    wire [HBW-1:0]    rmin_ext = HBW'(rmin);
    wire [HBW-1:0]    rmax_ext = HBW'(rmax);
    wire [8*2*HB-1:0] qhdr     = {rmax_ext, rmin_ext};
    reg [DW-1:0]      drmin;
    reg [7:0]         pk;

    // ------------------------------------------------------ handshakes
    assign in_ready = (state == S_IDLE) || (state == S_ARG) || (state == S_CSRW_A) ||
                      (state == S_CSRW_D) || (state == S_CSRR_A) || (state == S_LDV) ||
                      (state == S_DHDR) || (state == S_DBYTE);
    wire in_fire  = in_valid && in_ready;
    wire obuf_free = !obuf_v || out_ready;

    assign out_data  = obuf;
    assign out_valid = obuf_v;
    assign busy      = state != S_IDLE;

    // --------------------------------------------------- selected phase inc
    wire [31:0] inc_sel = (layer == 2'd0) ? inc0 : (layer == 2'd1) ? inc1 : inc2;

    // ------------------------------------------------------ pair addressing
    wire [L-1:0] k2    = {k[L-2:0], 1'b0};
    wire [L-1:0] i_adj = k2;
    wire [L-1:0] i_sh  = {k[L-2:0], 1'b1};
    reg  [L-1:0] pi, pj;
    reg          pvalid;
    reg  [L-1:0] kmax;

    always @(*) begin
        pvalid = 1'b1;
        if (uop == U_ROT && layer == 2'd1) begin
            pi = i_sh;
            pj = i_sh + 1'b1;
            kmax = KMAX_SH;
        end else if (uop == U_ROT && layer == 2'd2) begin
            pi = k;
            pj = k + S3V;
            pvalid = ~k[S3B];
            kmax = KMAX_L3;
        end else begin
            pi = i_adj;
            pj = i_adj + 1'b1;
            kmax = KMAX_ADJ;
        end
    end

    // --------------------------------------------------------- register file
    reg  [L-1:0]         ra, rb;
    wire signed [DW-1:0] rda, rdb;
    reg                  wea, web;
    reg  [L-1:0]         wa, wb;
    reg  signed [DW-1:0] wda, wdb;

    nqx_s1_rf u_rf (
        .clk(clk), .rst_n(rst_n), .ra(ra), .rb(rb), .rda(rda), .rdb(rdb),
        .wea(wea), .wa(wa), .wda(wda), .web(web), .wb(wb), .wdb(wdb)
    );

    // ---------------------------------------------------------------- CORDIC
    wire [ZW-1:0] zr    = acc[31:32-ZW] + {{(ZW-1){1'b0}}, acc[31-ZW]};
    wire [ZW-1:0] zphase = inv ? -zr : zr;

    wire                 c_ready;
    wire                 c_in_valid = (state == S_ISSUE) && pvalid && c_ready;
    wire                 c_in_vec   = uop == U_POLAR;
    wire signed [DW-1:0] c_in_y     = (uop == U_IPOLAR) ? {DW{1'b0}} : rdb;
    wire        [ZW-1:0] c_in_z     = (uop == U_ROT) ? zphase :
                                      (uop == U_IPOLAR) ? rdb[ZW-1:0] : {ZW{1'b0}};

    wire                 c_out_valid;
    wire signed [DW-1:0] c_out_x, c_out_y;
    wire        [ZW-1:0] c_out_z;
    wire                 c_out_sat;
    wire [2*L-1:0]       c_out_tag;
    wire                 c_busy;

    generate
        if (ITER_CORDIC != 0) begin : g_cordic_iter
            /* verilator lint_off UNUSEDSIGNAL */
            wire it_busy;
            /* verilator lint_on UNUSEDSIGNAL */
            nqx_s1_cordic_iter #(.TAG_W(2*L)) u_cordic (
                .clk(clk), .rst_n(rst_n),
                .in_valid(c_in_valid), .in_ready(c_ready), .in_vec(c_in_vec),
                .in_x(rda), .in_y(c_in_y), .in_z(c_in_z), .in_tag({pi, pj}),
                .out_valid(c_out_valid), .out_x(c_out_x), .out_y(c_out_y), .out_z(c_out_z),
                .out_sat(c_out_sat), .out_tag(c_out_tag), .busy(it_busy)
            );
        end else begin : g_cordic_pipe
            assign c_ready = 1'b1;
            nqx_s1_cordic #(.TAG_W(2*L)) u_cordic (
                .clk(clk), .rst_n(rst_n),
                .in_valid(c_in_valid), .in_vec(c_in_vec), .in_x(rda), .in_y(c_in_y),
                .in_z(c_in_z), .in_tag({pi, pj}),
                .out_valid(c_out_valid), .out_x(c_out_x), .out_y(c_out_y), .out_z(c_out_z),
                .out_sat(c_out_sat), .out_tag(c_out_tag)
            );
        end
    endgenerate

    // Pipeline occupancy for the drain phase.
    reg [5:0] inflight;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            inflight <= 6'd0;
        else
            inflight <= inflight + {5'd0, c_in_valid} - {5'd0, c_out_valid};
    end
    assign c_busy = (inflight != 6'd0);

    // ------------------------------------------------------------- quantizer
    wire [2:0] rq;
    wire       rs;
    wire [2:0] tq3;
    wire       ts;

    nqx_s1_rcode u_rcode (.d(qd), .rng(qrng), .q(rq));
    nqx_s1_rsign u_rsign (.d(qd), .rng(qrng), .q(qq), .s(rs));
    nqx_s1_tcode u_tcode (.t(qt), .q(tq3), .s(ts));

    // ----------------------------------------------------------- dequantizer
    localparam [DW-1:0] RLIM = (1 << (DW - 1)) - 1;

    wire [DW-1:0] h_rmin = hdr[DW-1:0];
    wire [DW-1:0] h_rmax = hdr[8*HB +: DW];
    wire          h_sat  = (h_rmin > RLIM) || (h_rmax > RLIM);
    wire [DW-1:0] c_rmin = (h_rmin > RLIM) ? RLIM : h_rmin;
    wire [DW-1:0] c_rmax = (h_rmax > RLIM) ? RLIM : h_rmax;
    wire [DW-1:0] c_rng  = (c_rmax > c_rmin) ? c_rmax - c_rmin : {DW{1'b0}};

    wire          div_start = state == S_DPREP;
    wire [SW-1:0] step;
    wire          div_busy;

    nqx_s1_div28 u_div (
        .clk(clk), .rst_n(rst_n), .start(div_start), .rng(c_rng), .step(step), .busy(div_busy)
    );

    wire signed [DW-1:0] dq_r;
    wire        [ZW-1:0] dq_t;
    wire                 dq_sat;

    nqx_s1_qdec u_qdec (
        .pk(pk), .rmin(drmin), .step(step), .refine(refine), .r(dq_r), .t(dq_t), .sat(dq_sat)
    );

    // ------------------------------------------------------ store rounding
    wire signed [DW:0]    st_rnd = {rda[DW-1], rda} + (1 << (FW - 1));
    /* verilator lint_off UNUSEDSIGNAL */
    wire signed [DW:0]    st_shr = st_rnd >>> FW;
    /* verilator lint_on UNUSEDSIGNAL */
    wire signed [DW-FW:0] st_val = st_shr[DW-FW:0];
    wire st_hi = st_val > 32767;
    wire st_lo = st_val < -32768;
    wire [15:0] st16 = st_hi ? 16'h7FFF : st_lo ? 16'h8000 : st_val[15:0];
    wire signed [HBW-1:0] st_raw = HBW'(rda);

    // ------------------------------------------------------- CSR read mux
    reg [31:0] csr_rdata;
    always @(*) begin
        case (caddr)
            CSR_ID:      csr_rdata = `NQX_CHIP_ID;
            CSR_VERSION: csr_rdata = `NQX_VERSION;
            CSR_PARAMS:  csr_rdata = `NQX_PARAMS_WORD;
            CSR_CTRL:    csr_rdata = {31'd0, refine};
            CSR_STATUS:  csr_rdata = {30'd0, st_sat, st_illegal};
            CSR_INC0:    csr_rdata = inc0;
            CSR_INC1:    csr_rdata = inc1;
            CSR_INC2:    csr_rdata = inc2;
            CSR_RMIN:    csr_rdata = {{(32-DW){1'b0}}, rmin};
            CSR_RMAX:    csr_rdata = {{(32-DW){1'b0}}, rmax};
            CSR_SCRATCH: csr_rdata = scratch;
            CSR_FEATURES: csr_rdata = {31'd0, ITER_CORDIC != 0};
            CSR_CYCLES:  csr_rdata = cnt_cycles;
            CSR_BUSY:    csr_rdata = cnt_busy;
            CSR_CORDIC:  csr_rdata = cnt_cordic;
            CSR_ENC:     csr_rdata = cnt_enc;
            CSR_DEC:     csr_rdata = cnt_dec;
            default:     csr_rdata = 32'd0;
        endcase
    end

    // --------------------------------------------- RF port and write muxing
    always @(*) begin
        ra  = pi;
        rb  = pj;
        wea = 1'b0;
        web = 1'b0;
        wa  = {L{1'b0}};
        wb  = {L{1'b0}};
        wda = {DW{1'b0}};
        wdb = {DW{1'b0}};
        case (state)
            S_STV:  ra = k;
            S_QRD: begin
                ra = k2;
                rb = k2 + 1'b1;
            end
            default: ;
        endcase
        if (c_out_valid) begin
            wea = 1'b1;
            wa  = c_out_tag[2*L-1:L];
            wda = c_out_x;
            web = 1'b1;
            wb  = c_out_tag[L-1:0];
            wdb = (uop == U_POLAR) ? {{(DW-ZW){c_out_z[ZW-1]}}, c_out_z} : c_out_y;
        end else if (state == S_LDV && in_fire && b[0]) begin
            wea = 1'b1;
            wa  = k;
            wda = {{(DW-16-FW){in_data[7]}}, in_data, lo, {FW{1'b0}}};
        end else if (state == S_DWR) begin
            wea = 1'b1;
            wa  = k2;
            wda = dq_r;
            web = 1'b1;
            wb  = k2 + 1'b1;
            wdb = {{(DW-ZW){dq_t[ZW-1]}}, dq_t};
        end
    end

    // --------------------------------------------------------- main FSM
    wire last_k = (k == kmax);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            uop        <= U_LDV;
            layer      <= 2'd0;
            inv        <= 1'b0;
            macro      <= M_NONE;
            mstep      <= 3'd0;
            k          <= {L{1'b0}};
            b          <= 3'd0;
            lo         <= 8'd0;
            acc        <= 32'd0;
            first      <= 1'b0;
            caddr      <= 8'd0;
            wshift     <= 32'd0;
            oshift     <= 32'd0;
            ocnt       <= 3'd0;
            obuf       <= 8'd0;
            obuf_v     <= 1'b0;
            inc0       <= `NQX_INC0_RESET;
            inc1       <= `NQX_INC1_RESET;
            inc2       <= `NQX_INC2_RESET;
            refine     <= 1'b1;
            st_illegal <= 1'b0;
            st_sat     <= 1'b0;
            rmin       <= {DW{1'b0}};
            rmax       <= {DW{1'b0}};
            scratch    <= 32'd0;
            cnt_cycles <= 32'd0;
            cnt_busy   <= 32'd0;
            cnt_cordic <= 32'd0;
            cnt_enc    <= 32'd0;
            cnt_dec    <= 32'd0;
            qd         <= {(DW+1){1'b0}};
            qt         <= {ZW{1'b0}};
            qq         <= 3'd0;
            qrng       <= {DW{1'b0}};
            hdr        <= {(16*HB){1'b0}};
            drmin      <= {DW{1'b0}};
            pk         <= 8'd0;
        end else begin
            cnt_cycles <= cnt_cycles + 1'b1;
            if (state != S_IDLE)
                cnt_busy <= cnt_busy + 1'b1;
            if (c_in_valid)
                cnt_cordic <= cnt_cordic + 1'b1;

            if (obuf_v && out_ready)
                obuf_v <= 1'b0;

            // CORDIC write-back side effects
            if (c_out_valid) begin
                if (c_out_sat)
                    st_sat <= 1'b1;
                if (uop == U_POLAR) begin
                    if (first) begin
                        rmin  <= c_out_x;
                        rmax  <= c_out_x;
                        first <= 1'b0;
                    end else begin
                        if (c_out_x < $signed(rmin)) rmin <= c_out_x;
                        if (c_out_x > $signed(rmax)) rmax <= c_out_x;
                    end
                end
            end

            case (state)
                // ------------------------------------------------ dispatch
                S_IDLE: if (in_fire) begin
                    macro <= M_NONE;
                    case (in_data)
                        OP_NOP: ;
                        OP_LDV:     begin uop <= U_LDV;     state <= S_LAUNCH; end
                        OP_STV:     begin uop <= U_STV;     state <= S_LAUNCH; end
                        OP_STVR:    begin uop <= U_STVR;    state <= S_LAUNCH; end
                        OP_POLAR:   begin uop <= U_POLAR;   state <= S_LAUNCH; end
                        OP_IPOLAR:  begin uop <= U_IPOLAR;  state <= S_LAUNCH; end
                        OP_QUANT:   begin uop <= U_QUANT;   state <= S_LAUNCH; end
                        OP_DEQUANT: begin uop <= U_DEQUANT; state <= S_LAUNCH; end
                        OP_GVNS:     begin inv <= 1'b0; state <= S_ARG; end
                        OP_GVNS_INV: begin inv <= 1'b1; state <= S_ARG; end
                        OP_ENC: begin
                            macro <= M_ENC;
                            mstep <= 3'd0;
                            {uop, layer, inv} <= macro_uop(M_ENC, 3'd0);
                            state <= S_LAUNCH;
                        end
                        OP_DEC: begin
                            macro <= M_DEC;
                            mstep <= 3'd0;
                            {uop, layer, inv} <= macro_uop(M_DEC, 3'd0);
                            state <= S_LAUNCH;
                        end
                        OP_SYNC: begin
                            oshift <= {24'd0, SYNC_BYTE};
                            ocnt   <= 3'd1;
                            state  <= S_EMIT;
                        end
                        OP_CSRW: state <= S_CSRW_A;
                        OP_CSRR: state <= S_CSRR_A;
                        default: st_illegal <= 1'b1;
                    endcase
                end

                S_ARG: if (in_fire) begin
                    if (in_data > 8'd2) begin
                        st_illegal <= 1'b1;
                        state      <= S_IDLE;
                    end else begin
                        layer <= in_data[1:0];
                        uop   <= U_ROT;
                        state <= S_LAUNCH;
                    end
                end

                // --------------------------------------------------- CSRs
                S_CSRW_A: if (in_fire) begin
                    caddr <= in_data;
                    b     <= 3'd0;
                    state <= S_CSRW_D;
                end

                S_CSRW_D: if (in_fire) begin
                    wshift <= wword;
                    b      <= b + 1'b1;
                    if (b == 3'd3) begin
                        case (caddr)
                            CSR_CTRL:    refine <= wword[0];
                            CSR_STATUS: begin
                                if (wword[0]) st_illegal <= 1'b0;
                                if (wword[1]) st_sat     <= 1'b0;
                            end
                            CSR_INC0:    inc0    <= wword;
                            CSR_INC1:    inc1    <= wword;
                            CSR_INC2:    inc2    <= wword;
                            CSR_SCRATCH: scratch <= wword;
                            default: ;
                        endcase
                        state <= S_IDLE;
                    end
                end

                S_CSRR_A: if (in_fire) begin
                    caddr <= in_data;
                    state <= S_CSRR_L;
                end

                S_CSRR_L: begin
                    oshift <= csr_rdata;
                    ocnt   <= 3'd4;
                    state  <= S_EMIT;
                end

                S_EMIT: if (obuf_free) begin
                    obuf   <= oshift[7:0];
                    obuf_v <= 1'b1;
                    oshift <= {8'd0, oshift[31:8]};
                    ocnt   <= ocnt - 1'b1;
                    if (ocnt == 3'd1)
                        state <= S_IDLE;
                end

                // ------------------------------------------- uop launcher
                S_LAUNCH: begin
                    k <= {L{1'b0}};
                    b <= 3'd0;
                    case (uop)
                        U_LDV:     state <= S_LDV;
                        U_STV,
                        U_STVR:    state <= S_STV;
                        U_ROT: begin
                            acc   <= inc_sel;
                            state <= (inc_sel == 32'd0) ? S_NEXT : S_ISSUE;
                        end
                        U_POLAR: begin
                            first <= 1'b1;
                            state <= S_ISSUE;
                        end
                        U_IPOLAR:  state <= S_ISSUE;
                        U_QUANT: begin
                            qrng  <= rmax - rmin;
                            state <= S_QHDR;
                        end
                        default:   state <= S_DHDR;
                    endcase
                end

                S_NEXT: begin
                    if (macro != M_NONE && mstep != 3'd5) begin
                        mstep <= mstep + 1'b1;
                        {uop, layer, inv} <= macro_uop(macro, mstep + 1'b1);
                        state <= S_LAUNCH;
                    end else begin
                        macro <= M_NONE;
                        state <= S_IDLE;
                    end
                end

                // ---------------------------------------------- load/store
                S_LDV: if (in_fire) begin
                    if (!b[0]) begin
                        lo <= in_data;
                        b  <= 3'd1;
                    end else begin
                        b <= 3'd0;
                        k <= k + 1'b1;
                        if (k == KLAST_EL)
                            state <= S_NEXT;
                    end
                end

                S_STV: if (obuf_free) begin
                    obuf_v <= 1'b1;
                    if (uop == U_STV) begin
                        obuf <= b[0] ? st16[15:8] : st16[7:0];
                        if (!b[0] && (st_hi || st_lo))
                            st_sat <= 1'b1;
                    end else begin
                        obuf <= st_raw[8*b +: 8];
                    end
                    if ((uop == U_STV && b == 3'd1) || (uop == U_STVR && b == HB_LAST)) begin
                        b <= 3'd0;
                        k <= k + 1'b1;
                        if (k == KLAST_EL)
                            state <= S_NEXT;
                    end else begin
                        b <= b + 1'b1;
                    end
                end

                // ------------------------------------------ CORDIC passes
                S_ISSUE: if (c_ready || !pvalid) begin
                    k   <= k + 1'b1;
                    acc <= acc + inc_sel;
                    if (last_k)
                        state <= S_DRAIN;
                end

                S_DRAIN: if (!c_busy && !c_out_valid)
                    state <= S_NEXT;

                // ------------------------------------------------ quantize
                S_QHDR: if (obuf_free) begin
                    obuf_v <= 1'b1;
                    obuf   <= qhdr[8*b +: 8];
                    b      <= b + 1'b1;
                    if (b == HDR_LAST)
                        state <= S_QRD;
                end

                S_QRD: begin
                    qd    <= $signed({rda[DW-1], rda}) - $signed({1'b0, rmin});
                    qt    <= rdb[ZW-1:0];
                    state <= S_QC1;
                end

                S_QC1: begin
                    qq    <= rq;
                    state <= S_QC2;
                end

                S_QC2: if (obuf_free) begin
                    obuf_v <= 1'b1;
                    obuf   <= {ts, tq3, rs, qq};
                    k      <= k + 1'b1;
                    if (k == KMAX_ADJ) begin
                        cnt_enc <= cnt_enc + 1'b1;
                        state   <= S_NEXT;
                    end else begin
                        state <= S_QRD;
                    end
                end

                // ---------------------------------------------- dequantize
                S_DHDR: if (in_fire) begin
                    hdr <= {in_data, hdr[8*2*HB-1:8]};
                    b   <= b + 1'b1;
                    if (b == HDR_LAST)
                        state <= S_DPREP;
                end

                S_DPREP: begin
                    drmin <= c_rmin;
                    if (h_sat)
                        st_sat <= 1'b1;
                    state <= S_DDIV;
                end

                S_DDIV: if (!div_busy)
                    state <= S_DBYTE;

                S_DBYTE: if (in_fire) begin
                    pk    <= in_data;
                    state <= S_DWR;
                end

                S_DWR: begin
                    if (dq_sat)
                        st_sat <= 1'b1;
                    k <= k + 1'b1;
                    if (k == KMAX_ADJ) begin
                        cnt_dec <= cnt_dec + 1'b1;
                        state   <= S_NEXT;
                    end else begin
                        state <= S_DBYTE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
