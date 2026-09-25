// Formal harness for nqx_s1_hsio: the chip side of both 4-phase handshakes
// and the valid/ready core interface, proven for every legal host behaviour.
//
// Host pins are modelled as free inputs sampled on the chip clock. That is
// conservative for asynchronous inputs: any waveform the host can produce is
// seen by the design as some sequence of per-cycle values, which the solver
// explores exhaustively.
`default_nettype none

module hsio_props (
    input wire       clk,
    input wire [7:0] in_bus,
    input wire       in_req,
    input wire       out_ack,
    input wire       s_ready,
    input wire [7:0] m_data,
    input wire       m_valid
);
    reg rst_n = 1'b0;
    always @(posedge clk) rst_n <= 1'b1;

    wire       in_ack, out_req, s_valid, m_ready;
    wire [7:0] out_bus, s_data;

    nqx_s1_hsio dut (
        .clk(clk), .rst_n(rst_n),
        .in_bus(in_bus), .in_req(in_req), .in_ack(in_ack),
        .out_bus(out_bus), .out_req(out_req), .out_ack(out_ack),
        .s_data(s_data), .s_valid(s_valid), .s_ready(s_ready),
        .m_data(m_data), .m_valid(m_valid), .m_ready(m_ready)
    );

    reg past_ok = 1'b0;
    always @(posedge clk) past_ok <= rst_n;

    // ------------------------------------------------ host assumptions
    always @(*) if (!past_ok) assume (!in_req && !out_ack && !m_valid);

    always @(posedge clk) if (past_ok) begin
        // in_req: rise only while in_ack is low, hold until in_ack, fall
        // only after in_ack, stay low until in_ack has fallen.
        if (!$past(in_req) && $past(in_ack)) assume (!in_req);
        if ($past(in_req) && !$past(in_ack)) assume (in_req);
        // Bundled data: in_bus stable while the request is outstanding.
        if ($past(in_req) && in_req && !$past(in_ack)) assume (in_bus == $past(in_bus));
        // out_ack: rise only while out_req is high, hold until out_req
        // falls, stay low until the next out_req.
        if (!$past(out_ack) && !$past(out_req)) assume (!out_ack);
        if ($past(out_ack) && $past(out_req)) assume (out_ack);
        // Core producer obeys valid/ready.
        if ($past(m_valid) && !$past(m_ready)) assume (m_valid && m_data == $past(m_data));
    end

    // -------------------------------------------------- chip obligations
    always @(posedge clk) if (past_ok) begin
        // in_ack rises only for a request and falls only after it is withdrawn.
        if (!$past(in_ack) && in_ack) assert ($past(in_req));
        if ($past(in_ack) && !in_ack) assert (!$past(in_req));
        // out_req rises only when the host is idle, holds until acknowledged,
        // and out_bus is stable while out_req is high.
        if (!$past(out_req) && out_req) assert (!$past(out_ack));
        if ($past(out_req) && !$past(out_ack)) assert (out_req);
        if ($past(out_req) && out_req) assert (out_bus == $past(out_bus));
        // Core consumer side obeys valid/ready.
        if ($past(s_valid) && !$past(s_ready)) assert (s_valid && s_data == $past(s_data));
    end

    // ------------------------------------------------ data integrity
    // Every byte offered by the host reaches the core once and unchanged.
    reg [7:0] held;
    reg       pending = 1'b0;
    always @(posedge clk) begin
        if (in_req && !in_ack && !pending) begin
            held    <= in_bus;
            pending <= 1'b1;
        end
        if (s_valid && s_ready) pending <= 1'b0;
    end
    always @(posedge clk) if (past_ok && s_valid) assert (pending && s_data == held);
    // Inductive strengthening (reachable-state invariants, also proven).
    always @(posedge clk) if (past_ok && pending && !s_valid && !in_ack)
        assert (in_req && held == in_bus);
    always @(posedge clk) if (past_ok && in_ack && !s_valid) assert (!pending);

    // Every core byte reaches the pins once and unchanged.
    reg [7:0] sent;
    reg       in_flight = 1'b0;
    reg       out_req_q = 1'b0;
    reg       cap_q = 1'b0;
    always @(posedge clk) begin
        out_req_q <= out_req;
        cap_q     <= m_valid && m_ready;
        if (out_req_q && !out_req) in_flight <= 1'b0;
        if (m_valid && m_ready) begin
            sent      <= m_data;
            in_flight <= 1'b1;
        end
    end
    always @(posedge clk) if (past_ok && out_req) assert (in_flight && out_bus == sent);
    always @(posedge clk) if (past_ok && in_flight) assert (out_bus == sent && !m_ready);
    always @(posedge clk) if (past_ok && in_flight) assert (cap_q || out_req || out_req_q);
    always @(posedge clk) if (past_ok && cap_q) assert (in_flight && !out_req);

    // -------------------------------------------------------- liveness
    // A cooperative host always makes progress (bounded response).
    reg [3:0] wait_in = 0;
    always @(posedge clk) wait_in <= (in_req && !in_ack && !s_valid) ? wait_in + 1'b1 : 4'd0;
    always @(posedge clk) if (past_ok) assert (wait_in < 4'd4);

    always @(posedge clk) cover (past_ok && out_req && out_ack && s_valid && s_ready);
endmodule

`default_nettype wire
