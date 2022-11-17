# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Cyril Koenig <cykoenig@iis.ee.ethz.ch>

# This constraint file is written for VCU128 + FMC XM105 Debug Card and is included only when EXT_JTAG = 1

# 5 MHz max JTAG
create_clock -period 200 -name jtag_tck_i_0 [get_pins test_dbg_vcu128_i/jtag_tck_i_0]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_pins jtag_tck_i_0_IBUF_inst/O]
set_property CLOCK_BUFFER_TYPE NONE [get_nets -of [get_pins jtag_tck_i_0_IBUF_inst/O]]

# Create asynchronous clock group between JTAG TCK and SoC clock.
set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins test_dbg_vcu128_i/jtag_tck_i_0]] -group [get_clocks -of_objects [get_pins test_dbg_vcu128_i/clk_wiz/clk* -filter {DIRECTION == "OUT"}]]

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