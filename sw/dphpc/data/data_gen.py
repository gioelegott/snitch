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

    A0 = gen_rand_csr_matrix(m=4, n=4, density=0.4)
    A1 = gen_rand_csr_matrix(m=4, n=4, density=0.4)
    FILTER0 = gen_rand_csr_matrix(m=2, n=2, density=0.4)
    FILTER1 = gen_rand_csr_matrix(m=2, n=2, density=0.4)

    A0_TEN = torch.from_numpy(A0.todense())
    A1_TEN = torch.from_numpy(A1.todense())
    A_TEN  = torch.stack((A0_TEN, A1_TEN), 0)

    F0_TEN = torch.from_numpy(FILTER0.todense())
    F1_TEN = torch.from_numpy(FILTER1.todense())
    F_TEN  = torch.stack((F0_TEN, F1_TEN), 0).unsqueeze(0)

    RES0_TEN = conv2d(ifmap=A_TEN, weights=F_TEN, padding=0, stride=1).squeeze(0)
    RES0 = sp.csr_matrix(RES0_TEN.detach().numpy())

    kwargs = {'name': 'conv2d_csr', 'A0': A0, 'A1': A1, 'FILTER0': FILTER0, 'FILTER1': FILTER1, 'RES0' : RES0}

    gen_data_header_file(args.outdir, args.tpl, **kwargs)


if __name__ == "__main__":
    main()
