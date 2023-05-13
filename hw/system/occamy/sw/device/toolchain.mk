# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

######################
# Invocation options #
######################

DEBUG ?= OFF # ON to turn on debugging symbols

###################
# Build variables #
###################

# Compiler toolchain
# CC      = riscv32-unknown-elf-gcc
# AR      = riscv32-unknown-elf-ar
# OBJCOPY = riscv32-unknown-elf-objcopy
# OBJDUMP = riscv32-unknown-elf-objdump
# READELF = riscv32-unknown-elf-readelf
RV_CC        = /usr/scratch2/rapanui/lbertaccini/snitch_occamy_vsum_test/riscv32-pulp-llvm-centos7-131/bin/clang
RV_LD        = /usr/scratch2/rapanui/lbertaccini/snitch_occamy_vsum_test/riscv32-pulp-llvm-centos7-131/bin/ld.lld
RV_AR        = /usr/scratch2/rapanui/lbertaccini/snitch_occamy_vsum_test/riscv32-pulp-llvm-centos7-131/bin/llvm-ar
RV_OBJCOPY   = /usr/scratch2/rapanui/lbertaccini/snitch_occamy_vsum_test/riscv32-pulp-llvm-centos7-131/bin/llvm-objcopy
RV_OBJDUMP   = /usr/scratch2/rapanui/lbertaccini/snitch_occamy_vsum_test/riscv32-pulp-llvm-centos7-131/bin/llvm-objdump
RV_DWARFDUMP = /usr/scratch2/rapanui/lbertaccini/snitch_occamy_vsum_test/riscv32-pulp-llvm-centos7-131/bin/llvm-dwarfdump

# Compiler flags
CFLAGS += $(addprefix -I,$(INCDIRS))
CFLAGS += -mcpu=snitch
CFLAGS += -menable-experimental-extensions
CFLAGS += -mabi=ilp32d
CFLAGS += -mcmodel=medany
#CFLAGS += -mno-fdiv
CFLAGS += -ffast-math
CFLAGS += -fno-builtin-printf
CFLAGS += -fno-common
CFLAGS += -O3
ifeq ($(DEBUG), ON)
CFLAGS += -g
endif

# Linker flags
LDFLAGS += -fuse-ld=$(RV_LD)

# Archiver flags
ARFLAGS = rcs
