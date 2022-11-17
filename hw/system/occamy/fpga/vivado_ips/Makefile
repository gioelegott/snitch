# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Nils Wistoff <nwistoff@iis.ee.ethz.ch>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
ROOT        := ${MKFILE_DIR}../../../../..
VIVADO      ?= vivado
BENDER_TARGETS := 
# Use an ILA to observe the (* mark_debug = "true" *) signals
DEBUG       ?= 0
# Use external JTAG chain instead of Xilinx internal chain
EXT_JTAG    ?= 0

ifeq ($(EXT_JTAG), 1)
	BENDER_TARGETS := $(BENDER_TARGETS) -t bscane
endif

include $(ROOT)/util/Makefrag

occamy_xilinx: define-sources.tcl
	${VIVADO} -mode batch -source occamy_xilinx.tcl -tclargs $(DEBUG) $(EXT_JTAG)

define-sources.tcl:
	${BENDER} script vivado $(BENDER_TARGETS) > $@

clean:
	rm -rf .Xil occamy_xilinx vivado* define-sources.tcl component.xml
