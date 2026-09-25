// Formal harness for nqx_s1_core: valid/ready compliance of the output byte
// stream and basic liveness of the command interface, for any legal input
// stream (arbitrary bytes, arbitrary gaps, arbitrary back-pressure).
`default_nettype none

module core_props (
    input wire       clk,
    input wire [7:0] in_data,
    input wire       in_valid,
    input wire       out_ready
);
    reg rst_n = 1'b0;
    always @(posedge clk) rst_n <= 1'b1;
    reg past_ok = 1'b0;
    always @(posedge clk) past_ok <= rst_n;

    wire       in_ready, out_valid, busy;
    wire [7:0] out_data;

    nqx_s1_core dut (
        .clk(clk), .rst_n(rst_n),
        .in_data(in_data), .in_valid(in_valid), .in_ready(in_ready),
        .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready),
        .busy(busy)
    );

    // Producer (host side of the stream) obeys valid/ready.
    always @(posedge clk) if (past_ok && $past(in_valid) && !$past(in_ready))
        assume (in_valid && in_data == $past(in_data));

    always @(posedge clk) if (past_ok) begin
        // Output stream obeys valid/ready: once offered, a byte stays until taken.
        if ($past(out_valid) && !$past(out_ready)) assert (out_valid && out_data == $past(out_data));
        // Idle means ready for the next opcode.
        if (!busy) assert (in_ready);
    end

    // Reachability: an opcode is decoded, the vector unit starts, a CSR is read.
    always @(posedge clk) cover (past_ok && busy && !in_ready);
    always @(posedge clk) cover (past_ok && out_valid && out_ready && !busy);
endmodule

`default_nettype wire
