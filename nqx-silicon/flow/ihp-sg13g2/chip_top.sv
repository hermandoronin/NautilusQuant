// NQX-S1 chip top for IHP SG13G2: 31 pads (23 signal, 8 power) around the
// nqx_s1_top digital block. Pad placement is set by PAD_* in config.yaml.
`default_nettype none

`ifdef USE_POWER_PINS
`define NQX_PWR_ONLY .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS)
`define NQX_PWR      .iovdd(IOVDD), .iovss(IOVSS), .vdd(VDD), .vss(VSS),
`else
`define NQX_PWR_ONLY
`define NQX_PWR
`endif

module chip_top (
`ifdef USE_POWER_PINS
    inout wire       VDD,
    inout wire       VSS,
    inout wire       IOVDD,
    inout wire       IOVSS,
`endif
    inout wire       clk_PAD,
    inout wire       rst_n_PAD,
    inout wire [7:0] in_bus_PAD,
    inout wire       in_req_PAD,
    inout wire       in_ack_PAD,
    inout wire [7:0] out_bus_PAD,
    inout wire       out_req_PAD,
    inout wire       out_ack_PAD,
    inout wire       busy_PAD
);
    wire       clk, rst_n, in_req, in_ack, out_req, out_ack, busy;
    wire [7:0] in_bus, out_bus;

    // Power pads: core VDD/VSS (1.2 V) and I/O IOVDD/IOVSS (3.3 V), two of each
    (* keep *) sg13g2_IOPadVdd    vdd_pad0   (`NQX_PWR_ONLY);
    (* keep *) sg13g2_IOPadVss    vss_pad0   (`NQX_PWR_ONLY);
    (* keep *) sg13g2_IOPadIOVdd  iovdd_pad0 (`NQX_PWR_ONLY);
    (* keep *) sg13g2_IOPadIOVss  iovss_pad0 (`NQX_PWR_ONLY);
    (* keep *) sg13g2_IOPadVdd    vdd_pad1   (`NQX_PWR_ONLY);
    (* keep *) sg13g2_IOPadVss    vss_pad1   (`NQX_PWR_ONLY);
    (* keep *) sg13g2_IOPadIOVdd  iovdd_pad1 (`NQX_PWR_ONLY);
    (* keep *) sg13g2_IOPadIOVss  iovss_pad1 (`NQX_PWR_ONLY);

    // Input pads
    sg13g2_IOPadIn     clk_pad        (`NQX_PWR .pad(clk_PAD), .p2c(clk));
    sg13g2_IOPadIn     rst_n_pad      (`NQX_PWR .pad(rst_n_PAD), .p2c(rst_n));
    sg13g2_IOPadIn     in_bus0_pad    (`NQX_PWR .pad(in_bus_PAD[0]), .p2c(in_bus[0]));
    sg13g2_IOPadIn     in_bus1_pad    (`NQX_PWR .pad(in_bus_PAD[1]), .p2c(in_bus[1]));
    sg13g2_IOPadIn     in_bus2_pad    (`NQX_PWR .pad(in_bus_PAD[2]), .p2c(in_bus[2]));
    sg13g2_IOPadIn     in_bus3_pad    (`NQX_PWR .pad(in_bus_PAD[3]), .p2c(in_bus[3]));
    sg13g2_IOPadIn     in_bus4_pad    (`NQX_PWR .pad(in_bus_PAD[4]), .p2c(in_bus[4]));
    sg13g2_IOPadIn     in_bus5_pad    (`NQX_PWR .pad(in_bus_PAD[5]), .p2c(in_bus[5]));
    sg13g2_IOPadIn     in_bus6_pad    (`NQX_PWR .pad(in_bus_PAD[6]), .p2c(in_bus[6]));
    sg13g2_IOPadIn     in_bus7_pad    (`NQX_PWR .pad(in_bus_PAD[7]), .p2c(in_bus[7]));
    sg13g2_IOPadIn     in_req_pad     (`NQX_PWR .pad(in_req_PAD), .p2c(in_req));
    sg13g2_IOPadIn     out_ack_pad    (`NQX_PWR .pad(out_ack_PAD), .p2c(out_ack));

    // Output pads, 4 mA
    sg13g2_IOPadOut4mA in_ack_pad     (`NQX_PWR .pad(in_ack_PAD), .c2p(in_ack));
    sg13g2_IOPadOut4mA out_bus0_pad   (`NQX_PWR .pad(out_bus_PAD[0]), .c2p(out_bus[0]));
    sg13g2_IOPadOut4mA out_bus1_pad   (`NQX_PWR .pad(out_bus_PAD[1]), .c2p(out_bus[1]));
    sg13g2_IOPadOut4mA out_bus2_pad   (`NQX_PWR .pad(out_bus_PAD[2]), .c2p(out_bus[2]));
    sg13g2_IOPadOut4mA out_bus3_pad   (`NQX_PWR .pad(out_bus_PAD[3]), .c2p(out_bus[3]));
    sg13g2_IOPadOut4mA out_bus4_pad   (`NQX_PWR .pad(out_bus_PAD[4]), .c2p(out_bus[4]));
    sg13g2_IOPadOut4mA out_bus5_pad   (`NQX_PWR .pad(out_bus_PAD[5]), .c2p(out_bus[5]));
    sg13g2_IOPadOut4mA out_bus6_pad   (`NQX_PWR .pad(out_bus_PAD[6]), .c2p(out_bus[6]));
    sg13g2_IOPadOut4mA out_bus7_pad   (`NQX_PWR .pad(out_bus_PAD[7]), .c2p(out_bus[7]));
    sg13g2_IOPadOut4mA out_req_pad    (`NQX_PWR .pad(out_req_PAD), .c2p(out_req));
    sg13g2_IOPadOut4mA busy_pad       (`NQX_PWR .pad(busy_PAD), .c2p(busy));

    nqx_s1_top u_nqx (
        .clk(clk), .rst_n(rst_n),
        .in_bus(in_bus), .in_req(in_req), .in_ack(in_ack),
        .out_bus(out_bus), .out_req(out_req), .out_ack(out_ack),
        .busy(busy)
    );
endmodule

`default_nettype wire
