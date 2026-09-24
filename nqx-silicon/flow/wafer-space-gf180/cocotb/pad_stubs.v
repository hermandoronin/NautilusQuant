// Behavioural stand-ins for the GF180MCU pad cells and wafer.space ID/logo IP,
// for simulating without a PDK (NQX_PAD_STUBS=1). Not for sign-off.
module gf180mcu_fd_io__dvdd(inout DVDD, DVSS, VDD, VSS); endmodule
module gf180mcu_fd_io__dvss(inout DVDD, DVSS, VDD, VSS); endmodule
module gf180mcu_fd_io__in_s(inout DVDD, DVSS, VDD, VSS, output Y, inout PAD, input PU, PD); assign Y = PAD; endmodule
module gf180mcu_fd_io__in_c(inout DVDD, DVSS, VDD, VSS, output Y, inout PAD, input PU, PD); assign Y = PAD; endmodule
module gf180mcu_fd_io__bi_24t(inout DVDD, DVSS, VDD, VSS, input A, OE, output Y, inout PAD, input CS, SL, IE, PU, PD); assign PAD = OE ? A : 1'bz; assign Y = IE ? PAD : 1'b0; endmodule
module gf180mcu_fd_io__asig_5p0(inout DVDD, DVSS, VDD, VSS, inout ASIG5V, inout PAD); endmodule
module gf180mcu_ws_ip__qrcode_id(); endmodule
module gf180mcu_ws_ip__shuttle_id(); endmodule
module gf180mcu_ws_ip__project_id(); endmodule
module gf180mcu_ws_ip__marker(); endmodule
module gf180mcu_ws_ip__logo(); endmodule
