#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Marco Bertuletti <mbertuletti@iis.ee.ethz.ch>

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

import argparse
import pathlib
from mako.template import Template


def gen_data_header_file(outdir: pathlib.Path.cwd(), tpl: pathlib.Path.cwd(), **kwargs):

    file = outdir / f"data_{kwargs['name']}.h"

    print(tpl, outdir, kwargs['name'])

    template = Template(filename=str(tpl))
    with file.open('w') as f:
        f.write(template.render(**kwargs))


def gen_rand_csr_matrix(m: int, n: int, density: float) -> sp.csr_matrix:
    return sp.random(m, n, density, format='csr')


def main():

    parser = argparse.ArgumentParser(description='Generate data for kernels')
    parser.add_argument(
        "-o",
        "--outdir",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        required=False,
        help='Select out directory of generated data files'
    )
    parser.add_argument(
        "-t",
        "--tpl",
        type=pathlib.Path,
        required=False,
        default=pathlib.Path.cwd() / "data_softmax_dense.h.tpl",
        help='Path to mako template'
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help='Set verbose'
    )
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        required=False,
        default=5,
        help='Matrix dimension'
    )

    parser.add_argument(
        "-a",
        "--axis",
        type=int,
        required=False,
        default=0,
        help='Softmax axis'
    )

    args = parser.parse_args()

    # Create sparse matrix
    n = args.dimension
    m = args.dimension
    ax = args.axis
    density = 0.1
    A = sp.random(m, n, density, format='csr')

    #Compute result
    A_dense = A.todense()
    A_dense = A_dense - np.max(np.array(A_dense), axis=ax, keepdims=True)
    print(A_dense)
    A_dense = tf.convert_to_tensor(A_dense)
    C_dense = tf.keras.activations.softmax(A_dense, axis=args.axis)
    #Convert result to sparse format
    C_dense = C_dense.numpy()
    print(C_dense)
    A_dense = A_dense.numpy()

    kwargs = {'name': 'softmax_dense', 'A': A_dense.reshape(-1), 'C': C_dense.reshape(-1), 'dim' : n}

    gen_data_header_file(args.outdir, args.tpl, **kwargs)


if __name__ == "__main__":
    main()
