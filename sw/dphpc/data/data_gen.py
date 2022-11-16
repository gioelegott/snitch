#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

import numpy as np
import scipy.sparse as sp
import argparse
import pathlib
from mako.template import Template

np.set_printoptions(precision=3)


def gen_data_header_file(outdir: pathlib.Path, tpl: pathlib.Path, **kwargs):

    file = outdir / f"data_{kwargs['name']}.h"

    template = Template(filename=str(tpl))
    with file.open('w') as f:
        f.write(template.render(**kwargs))


def gen_rand_csr_matrix(m: int, n: int, density: float) -> sp.csr_matrix:
    return sp.random(m, n, density, format='csr')


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

    A = gen_rand_csr_matrix(m=32, n=32, density=0.1)
    B = gen_rand_csr_matrix(m=32, n=32, density=0.1)
    C = A * B
    C.sort_indices()

    kwargs = {'name': 'matmul_csr', 'A': A, 'B': B, 'C': C}

    gen_data_header_file(args.outdir, args.tpl, **kwargs)


if __name__ == "__main__":
    main()
