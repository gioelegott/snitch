# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

# Allow benchmark to be built as a standalone library.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
    set(CMAKE_TOOLCHAIN_FILE toolchain-llvm CACHE STRING "Toolchain to use")
    set(SNITCH_SIMULATOR ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/system/snitch_cluster/bin/snitch_cluster.vlt CACHE PATH "")

    project(DPHPC LANGUAGES C ASM)
    include(SnitchUtilities)

    add_compile_options(-O3 -g -ffunction-sections)

    # Build the runtime.
    add_subdirectory(../snRuntime snRuntime)
endif()

macro(add_snitch_dphpc_executable name)
    add_executable(${ARGV})
    target_link_libraries(${name} ${SNITCH_RUNTIME})
    target_link_options(${name} PRIVATE "SHELL:-T ${LINKER_SCRIPT}")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_LIST_DIR}/data/data_${name}.h
        COMMAND ./data/data_gen_${name}.py --outdir ${CMAKE_CURRENT_LIST_DIR}/data
        DEPENDS data/data_gen_${name}.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    )
    add_custom_command(
        TARGET ${name}
        POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -dhS $<TARGET_FILE:${name}> > $<TARGET_FILE:${name}>.s)
    # Run target for banshee
    if (SNITCH_RUNTIME STREQUAL "snRuntime-banshee")
        add_custom_target( run-banshee-${name}
            COMMAND ${SNITCH_BANSHEE} --no-opt-llvm --no-opt-jit --configuration ${CMAKE_CURRENT_LIST_DIR}/../banshee/config/snitch_cluster.yaml --trace $<TARGET_FILE:${name}> > $<TARGET_FILE:${name}>.trace
            COMMAND cat $<TARGET_FILE:${name}>.trace | ${SPIKE_DASM} > $<TARGET_FILE:${name}>.trace.txt
            COMMAND awk -F\" \" '{print>\"${name}\"$$3\".txt\"}' $<TARGET_FILE:${name}>.trace.txt
            DEPENDS $<TARGET_FILE:${name}>)
    endif()
    # Run target for RTL simulator
    if (SNITCH_RUNTIME STREQUAL "snRuntime-cluster")
        add_custom_target( run-rtl-${name}
            COMMAND ${SNITCH_SIMULATOR} $<TARGET_FILE:${name}>
            COMMAND for f in logs/trace_hart_*.dasm\; do ${SPIKE_DASM} < $$f | ${PYTHON} ${SNRUNTIME_SRC_DIR}/../../util/gen_trace.py > $$\(echo $$f | sed 's/\\.dasm/\\.txt/'\)\; done
            COMMAND ${CMAKE_CURRENT_LIST_DIR}/perf_extr.py --nproc 8 --section 1 --output ${name}_perf.csv
            DEPENDS $<TARGET_FILE:${name}>)
    endif()
    target_link_libraries(${name} kernels utils)
endmacro()

enable_testing()

include_directories(include data src/utils src/kernels)
include_directories(${SNRUNTIME_INCLUDE_DIRS})

add_library(kernels src/kernel/matmul_csr.c src/kernel/softmax_csr.c src/kernel/softmax_dense.c)
add_library(utils src/utils/utils.c)

target_link_libraries(kernels ${SNITCH_RUNTIME} utils)
target_link_libraries(utils ${SNITCH_RUNTIME})

add_snitch_dphpc_executable(matmul_csr src/main_matmul_csr.c data/data_matmul_csr.h)
add_snitch_dphpc_executable(softmax_csr src/main_softmax_csr.c data/data_softmax_csr.h)
add_snitch_dphpc_executable(softmax_dense src/main_softmax_dense.c data/data_softmax_dense.h)

set(SNITCH_TEST_PREFIX snDPHPC-)

add_snitch_raw_test_args(matmul_csr matmul_csr --configuration ${CMAKE_CURRENT_LIST_DIR}/../banshee/config/snitch_cluster.yaml)

add_snitch_raw_test_args(softmax_csr softmax_csr --configuration ${CMAKE_CURRENT_LIST_DIR}/../banshee/config/snitch_cluster.yaml)

add_snitch_raw_test_args(softmax_dense softmax_dense --configuration ${CMAKE_CURRENT_LIST_DIR}/../banshee/config/snitch_cluster.yaml)
