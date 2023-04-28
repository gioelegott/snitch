#!/usr/bin/env python3
# Copyright 2020 ETH Zurich and University of Bologna.
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51
#
# Paul Scheffler <paulsc@iis.ee.ethz.ch>
#
# This class implements a minimal wrapping IPC server for `tb_lib`.
# `__main__` shows a demonstrator for it, running a simulation and accessing its memory.

import os
import sys
import tempfile
import subprocess
import struct
from time import sleep
import csv
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
import array
import numpy as np
import golden_models
import math

class VerifySim:

    def __init__(self, sim_bin: str, snitch_bin: str):
        self.sim_bin = sim_bin
        self.snitch_bin = snitch_bin
        self.sim = None
        self.tmpdir = None

    def start(self):
        # Create FIFOs
        self.tmpdir = tempfile.TemporaryDirectory() # creates a directory for the Fifo that gets deleted after the programm is done

        tx_fd = os.path.join(self.tmpdir.name, 'tx') # path to tx file
        os.mkfifo(tx_fd) # creates a FIFO

        rx_fd = os.path.join(self.tmpdir.name, 'rx') # path to rx file
        os.mkfifo(rx_fd)

        # Start simulator process
        ipc_arg = f'--ipc,{tx_fd},{rx_fd}'
        verification_arg = f'--verification'
        self.sim = subprocess.Popen([self.sim_bin, self.snitch_bin, ipc_arg])
        
        # Open FIFOs
        self.tx = open(tx_fd, 'wb')
        self.rx = open(rx_fd, 'rb')

    def __sim_active(func):
        def inner(self, *args, **kwargs):
            if self.sim is None:
                raise RuntimeError(f'Snitch is not running (simulation `{self.sim_bin}`, binary `{self.snitch_bin}`)')
            return func(self, *args, **kwargs)
        return inner

    @__sim_active
    def read(self, addr: int, length: int) -> bytes:
        op = struct.pack('QQQ', 0, addr, length)
        self.tx.write(op)
        self.tx.flush()
        return self.rx.read(length)

    @__sim_active
    def write(self, addr: int, data: bytes):
        op = struct.pack('QQQ', 1, addr, len(data))
        self.tx.write(op)
        self.tx.write(data)
        self.tx.flush()

    @__sim_active
    def poll(self, addr: int, mask32: int, exp32: int):
        # TODO: check endiannesses
        op = struct.pack('QQLL', 2, addr, mask32, exp32)
        self.tx.write(op)
        self.tx.flush()
        return int.from_bytes(self.rx.read(4))

    @__sim_active
    def ping(self, addr: int, length: int):
        op = struct.pack('QQQ', 3, addr, length)
        self.tx.write(op)
        self.tx.flush()

    # Simulator can exit only once TX FIFO closes
    @__sim_active
    def finish(self, wait_for_sim: bool = True):
        self.rx.close()
        self.tx.close()
        if (wait_for_sim):
            self.sim.wait()
        else:
            self.sim.terminate()
        self.tmpdir.cleanup()
        self.sim = None

# Read out address of kernel result and size out of binary dump file
def parse_elf(kernel_path, result_name):
    ''' from https://github.com/eliben/pyelftools/blob/master/examples/elf_low_high_api.py '''
    
    st_value = None
    st_size = None
    with open(kernel_path, 'rb') as stream:
        elffile = ELFFile(stream)
        section = elffile.get_section_by_name('.symtab')
        if not section:
            print('  No symbol table found. Perhaps this ELF has been stripped?')

        if isinstance(section, SymbolTableSection):
            y = section.get_symbol_by_name(result_name)
            for i in range(len(y)):
                st_value = y[i].entry["st_value"] # symbol table: value
                st_size = y[i].entry["st_size"] # symbol table: size
                return st_value, st_size;


# Compare the kernel result with the Golden Model
def compare(gm_result, kernel_result, correct):
    rel_tol=1e-6
    abs_tol=1e-6
    if gm_result.ndim == 1:
        # loop through 1d array
        for i in range(gm_result.shape[0]):
            isclose = math.isclose(kernel_result[i], gm_result[i], rel_tol=rel_tol, abs_tol=abs_tol)
            if not isclose:
                correct = False # If one entry is wrong, kernel result is wrong
                print("Relative error for entry [", i, "] is: ", abs((kernel_result[i] - gm_result[i])/max(kernel_result[i], gm_result[i])), ", Absolute error is", abs(kernel_result[i]-gm_result[i]) ,", relative tolerance is: ", rel_tol, " and absolute tolerance is: ", abs_tol)

    elif gm_result.ndim == 2:
        # loop through 2d array
        for i in range(gm_result.shape[0]):
            for j in range(gm_result.shape[1]):
                isclose = math.isclose(kernel_result[i][j], gm_result[i][j], rel_tol=rel_tol, abs_tol=abs_tol)
                if not isclose:
                    correct = False # If one entry is wrong, kernel result is wrong
                    print("Relative error for entry [", i, "][", j, "] (GM: ",gm_result[i][j], ", Occamy: ", kernel_result[i][j],") is: ", abs((kernel_result[i][j] - gm_result[i][j])/max(kernel_result[i][j], gm_result[i][j])), ", Absolute error is", abs(kernel_result[i][j]-gm_result[i][j]) ,", relative tolerance is ", rel_tol, " and absolute tolerance is: ", abs_tol)
    return correct

if __name__ == "__main__":
    sim = VerifySim(*sys.argv[1:])
    sim.start()
    
    # wait until simulation/binary is finished like in tb_bin.sv file (while ((exit_code = fesvr_tick()) == 0))
    tstr = b'simulation terminated'
    rstr = b' '
    print("VerifySim: Waiting for the simulation to end.\n")
    while tstr != rstr:
        rstr = sim.rx.read(len(tstr)) # len(): number of characters in string = 21 (without)
        sleep(1)

    # Run Golden Model of the kernel
    print("VerifySim: Running Golden Model\n")
    # Get path to the kernel's ELF and header file
    kernel_path = sys.argv[2]
    kernel_name = kernel_path.split(os.sep)[-3]
    kernel = getattr(golden_models, kernel_name)
    header_path = "sw/host/apps/"+kernel_name+"/data/data.h"
    gm_result, variable_name = kernel(header_path)
    
    # Read out DRAM
    print("VerifySim: Reading out DRAM\n")
    # check if kernel has one or multiple output variables and read them from memory
    if type(variable_name) is list:
        kernel_result = []
        i = 0
        for var in variable_name:
            # Read out address of kernel result and size out of binary file
            st_value, st_size = parse_elf(kernel_path, var)
            result = sim.read(st_value, st_size)
            result_double = array.array('d', result) # takes 8 bytes (64bit for doubles) and converts it to a double
            result_double = np.array(result_double, dtype= np.double)
            result_double = np.reshape(result_double, gm_result[i].shape)
            kernel_result.append(result_double)
            i += 1

    else: 
        # Read out address of kernel result and size out of binary file
        st_value, st_size = parse_elf(kernel_path, variable_name)
        result = sim.read(st_value, st_size)
        kernel_result = array.array('d', result) # takes 8 bytes (64bit for doubles) and converts it to a double
        kernel_result = np.array(kernel_result, dtype= np.double)
        kernel_result = np.reshape(kernel_result, gm_result.shape)

    # send confirmation to simulation
    sim.ping(0, 0) # address and string don't matter 

    sim.finish(wait_for_sim=True)


    # Compare result with golden model

    if type(variable_name) is list: 
        # kernel has multiple outputs
        if len(gm_result) != len(kernel_result):
            print("Error: Golden Model has different number of outputs than expected.")
        correct = True
        for j in range(len(kernel_result)): 
            if gm_result[j].ndim == 1:
                print("Golden Model result for output", variable_name[j] ,":")
                print(np.array(["{:0.16f}".format(x) for x in gm_result[j]]))
                print("Kernel result for output", variable_name[j] , ":")
                print(np.array(["{:0.16f}".format(x) for x in kernel_result[j]]))
            elif gm_result[j].ndim == 2:
                print("Golden Model result for output", variable_name[j] ,":")
                print(np.array([["{:0.16f}".format(x) for x in row] for row in gm_result[j]]))
                print("Kernel result for output", variable_name[j] , ":")
                print(np.array([["{:0.16f}".format(x) for x in row] for row in kernel_result[j]]))
            correct = compare(gm_result[j], kernel_result[j], correct)
        if correct:
            print("Kernel result is correct")
        else:
            print("Kernel result is wrong")

    else: 
        # kernel has a single output
        if gm_result.ndim == 1:
            print("Golden Model result:")
            print(np.array(["{:0.16f}".format(x) for x in gm_result]))
            print("Kernel result:")
            print(np.array(["{:0.16f}".format(x) for x in kernel_result]))
        elif gm_result.ndim == 2:
            print("Golden Model result:")
            print(np.array([["{:0.16f}".format(x) for x in row] for row in gm_result]))
            print("Kernel result:")
            print(np.array([["{:0.16f}".format(x) for x in row] for row in kernel_result]))
        correct = True
        correct = compare(gm_result, kernel_result, correct)
        if correct:
            print("Kernel result is correct")
        else:
            print("Kernel result is wrong")
