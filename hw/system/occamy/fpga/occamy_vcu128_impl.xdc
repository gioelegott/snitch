# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

set_property PACKAGE_PIN BJ51 [get_ports clk_100MHz_n]
set_property IOSTANDARD DIFF_SSTL12 [get_ports clk_100MHz_n]
set_property PACKAGE_PIN BH51 [get_ports clk_100MHz_p]
set_property IOSTANDARD DIFF_SSTL12 [get_ports clk_100MHz_p]

set_property PACKAGE_PIN BP26 [get_ports uart_rx_i_0]
set_property IOSTANDARD LVCMOS18 [get_ports uart_rx_i_0]
set_property PACKAGE_PIN BN26 [get_ports uart_tx_o_0]
set_property IOSTANDARD LVCMOS18 [get_ports uart_tx_o_0]

# Set RTC as false path
set_false_path -to occamy_vcu128_i/occamy_xilinx_0/inst/i_occamy/i_clint/i_sync_edge/i_sync/reg_q_reg[0]/D

if { $DEBUG } {
  # 5 MHz max JTAG
  create_clock -period 200 -name jtag_tck_i_0 [get_pins occamy_vcu128_i/jtag_tck_i_0]
  set_property CLOCK_DEDICATED_ROUTE FALSE [get_pins jtag_tck_i_0_IBUF_inst/O]
  set_property CLOCK_BUFFER_TYPE NONE [get_nets -of [get_pins jtag_tck_i_0_IBUF_inst/O]]
  
  # Create asynchronous clock group between JTAG TCK and SoC clock.
  set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins occamy_vcu128_i/jtag_tck_i_0]] -group [get_clocks -of_objects [get_pins occamy_vcu128_i/clk_wiz/clk* -filter {DIRECTION == "OUT"}]]
  
  # B23 - C14 (FMCP_HSPC_LA10_P) - J1.02 - VDD
  set_property PACKAGE_PIN B23     [get_ports jtag_vdd_o_0]
  set_property IOSTANDARD LVCMOS18 [get_ports jtag_vdd_o_0]
  # A23 - C15 (FMCP_HSPC_LA10_N) - J1.04 - GND
  set_property PACKAGE_PIN A23     [get_ports jtag_gnd_o_0]
  set_property IOSTANDARD LVCMOS18 [get_ports jtag_gnd_o_0]
  # B26 - H16 (FMCP_HSPC_LA11_P) - J1.06 - TCK
  set_property PACKAGE_PIN B26     [get_ports jtag_tck_i_0]
  set_property IOSTANDARD LVCMOS18 [get_ports jtag_tck_i_0]
  # B25 - H17 (FMCP_HSPC_LA11_N) - J1.08 - TDO
  set_property PACKAGE_PIN B25     [get_ports jtag_tdo_o_0]
  set_property IOSTANDARD LVCMOS18 [get_ports jtag_tdo_o_0]
  # J22 - G15 (FMCP_HSPC_LA12_P) - J1.10 - TDI
  set_property PACKAGE_PIN J22     [get_ports jtag_tdi_i_0]
  set_property IOSTANDARD LVCMOS18 [get_ports jtag_tdi_i_0]
  # H22 - G16 (FMCP_HSPC_LA12_N) - J1.12 - TNS
  set_property PACKAGE_PIN H22     [get_ports jtag_tms_i_0]
  set_property IOSTANDARD LVCMOS18 [get_ports jtag_tms_i_0]
  # A24 - D18 (FMCP_HSPC_LA13_N) - J1.16 - TRST (Jumper cable)
  set_property PACKAGE_PIN A24     [get_ports jtag_trst_ni_0]
  set_property IOSTANDARD LVCMOS18 [get_ports jtag_trst_ni_0]
  # B17 - H23 (FMCP_HSPC_LA19_N) - J1.40 - VDD (Jumper cable)
  set_property PACKAGE_PIN B17     [get_ports jtag_vdd_o_1]
  set_property IOSTANDARD LVCMOS18 [get_ports jtag_vdd_o_1]
  # D26 - D15 (FMCP_HSPC_LA09_N) - J1.39 - GND (Jumper cable)
  set_property PACKAGE_PIN D26     [get_ports jtag_gnd_o_1]
  set_property IOSTANDARD LVCMOS18 [get_ports jtag_gnd_o_1]
}