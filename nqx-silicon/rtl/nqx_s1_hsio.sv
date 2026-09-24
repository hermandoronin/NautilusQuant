// Asynchronous 4-phase (return-to-zero) byte handshake <-> valid/ready stream.
//
// Host -> chip:  host drives in_bus and raises in_req; chip samples the bus,
//                raises in_ack; host drops in_req; chip drops in_ack.
// Chip -> host:  chip drives out_bus and raises out_req; host reads the bus,
//                raises out_ack; chip drops out_req; host drops out_ack.
//
// in_req and out_ack are asynchronous to clk and pass through two-flop
// synchronizers. in_bus is stable whenever in_req is high (bundled-data
// rule), so it is sampled without a synchronizer. out_bus is driven one clock
// before out_req rises, so the data pins settle before the request.
`default_nettype none

module nqx_s1_hsio (
    input  wire       clk,
    input  wire       rst_n,
    // pins
    input  wire [7:0] in_bus,
    input  wire       in_req,
    output reg        in_ack,
    output reg  [7:0] out_bus,
    output reg        out_req,
    input  wire       out_ack,
    // core side
    output reg  [7:0] s_data,
    output reg        s_valid,
    input  wire       s_ready,
    input  wire [7:0] m_data,
    input  wire       m_valid,
    output wire       m_ready
);
    reg [1:0] req_sync;
    reg [1:0] ack_sync;
    wire req_s = req_sync[1];
    wire ack_s = ack_sync[1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            req_sync <= 2'b00;
            ack_sync <= 2'b00;
        end else begin
            req_sync <= {req_sync[0], in_req};
            ack_sync <= {ack_sync[0], out_ack};
        end
    end

    // ---------------------------------------------------------- inbound
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            in_ack  <= 1'b0;
            s_valid <= 1'b0;
            s_data  <= 8'd0;
        end else begin
            if (s_valid && s_ready)
                s_valid <= 1'b0;
            if (req_s && !in_ack && !s_valid) begin
                s_data  <= in_bus;
                s_valid <= 1'b1;
                in_ack  <= 1'b1;
            end else if (!req_s && in_ack) begin
                in_ack <= 1'b0;
            end
        end
    end

    // --------------------------------------------------------- outbound
    reg out_pend;
    assign m_ready = !out_req && !out_pend && !ack_s;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_req  <= 1'b0;
            out_pend <= 1'b0;
            out_bus  <= 8'd0;
        end else if (out_pend) begin
            out_pend <= 1'b0;
            out_req  <= 1'b1;
        end else if (out_req) begin
            if (ack_s)
                out_req <= 1'b0;
        end else if (m_valid && !ack_s) begin
            out_bus  <= m_data;
            out_pend <= 1'b1;
        end
    end
endmodule

`default_nettype wire
