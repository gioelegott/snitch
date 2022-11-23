#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

import numpy as np
import scipy.sparse as sp
import argparse
import pathlib
import torch
import torch.nn as nn
from mako.template import Template

np.set_printoptions(precision=3)


def gen_data_header_file(outdir: pathlib.Path, tpl: pathlib.Path, **kwargs):

    file = outdir / f"data_{kwargs['name']}.h"

    template = Template(filename=str(tpl))
    with file.open('w') as f:
        f.write(template.render(**kwargs))


def gen_rand_csr_matrix(m: int, n: int, density: float) -> sp.csr_matrix:
    return sp.random(m, n, density, format='csr')

def conv2d(ifmap, weights, padding=1, stride=1):
    n = 1
    ci, ih, iw = ifmap.shape
    co, _,fh, fw = weights.shape

    conv2d = nn.Conv2d(ci, co, (fh, fw), padding=((fh-1)//2, (fw-1)//2))
    conv2d.weight = nn.Parameter(weights, requires_grad=False)
    conv2d.bias = nn.Parameter(torch.zeros_like(conv2d.bias, dtype=weights.dtype), requires_grad=False)
    ofmap = conv2d(ifmap)

    return ofmap

def main():

    script_path = pathlib.Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description='Generate data for kernels')
    parser.add_argument(
        "-o",
        "--outdir",
        type=pathlib.Path,
        default=script_path,
        required=False,
        help='Select out directory of generated data files'
    )
    parser.add_argument(
        "-t",
        "--tpl",
        type=pathlib.Path,
        required=False,
        default=script_path / "data.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help='Set verbose'
    )

    args = parser.parse_args()

    ###############################
    ######## Matrices Gen #########
    ###############################
    channel_size = 2
    
    A = []
    for i in range(channel_size):
        A.append(gen_rand_csr_matrix(m=4, n=4, density=0.4))

    FILTER = []
    for j in range(channel_size):
        FILTER.append(gen_rand_csr_matrix(m=2, n=2, density=0.4))

    ###############################
    ######## Golden Model #########
    ###############################
    A_torch = {}
    A_TEN = []
    for i in range(channel_size):
        A_torch[i] = torch.from_numpy(A[i].todense())
        A_TEN.append(A_torch[i])
    A_TEN = torch.stack(A_TEN, 0)

    F_torch = {}
    F_TEN = []
    for j in range(channel_size):
        F_torch[j] = torch.from_numpy(FILTER[j].todense())
        F_TEN.append(F_torch[j])
    F_TEN = torch.stack(F_TEN, 0).unsqueeze(0)

    RES0_TEN = conv2d(ifmap=A_TEN, weights=F_TEN, padding=0, stride=1).squeeze(0)
    RES0 = sp.csr_matrix(RES0_TEN.detach().numpy())

    ###############################
    ######## Output  File #########
    ###############################
    kwargs = {'name': 'conv2d_csr', 'A' : A, 'FILTER': FILTER, 'RES0' : RES0, 'channel_size' : channel_size}

    gen_data_header_file(args.outdir, args.tpl, **kwargs)


if __name__ == "__main__":
    main()
