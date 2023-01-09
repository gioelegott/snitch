#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

import argparse
import pathlib
import subprocess
import logging
import glob

import pandas as pd
import numpy as np
from scipy import stats

# Log to measurements.log
logging.basicConfig(filename='measurements.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')


def read_run_from_csv(indir: pathlib.Path, binary: str, nproc: int, size: int, metric: str = 'cycles', stat: str = 'mean'):
    data = pd.DataFrame()
    # Read in csv_files
    for s in size:
        glob_str = f"{indir}/{binary}_n{nproc}_s{s}_r*.csv"
        files = glob.glob(glob_str)
        values = []
        for f in files:
            values.append(pd.read_csv(f, index_col=0).loc[stat, metric])
        data[s] = values
    return data


def mean_confidence_interval(data, confidence):
    m = data.apply(np.mean, axis=0)
    se = data.apply(stats.sem)
    n = len(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, se, h


def run_measurements(outdir: pathlib.Path, test_config: list, nproc: int):

    # Create outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for cfg in test_config:
        for n in cfg['nproc']:
            for s in cfg['size']:
                results = []
                if cfg['num_runs'] < nproc:
                    nproc = cfg['num_runs']
                logging.info(f"Parallelizing {cfg['num_runs']} with {nproc} processes")
                for i in range(0, cfg['num_runs'], nproc):
                    # Run measurement
                    logging.info(f"Running measurement {i+1}-{i+nproc}/{cfg['num_runs']} of {cfg['binary']} with {n} cores and size {s}")
                    # Remove build folders
                    cmd = "rm -rf build*"
                    print(cmd)
                    subprocess.run(cmd, shell=True, check=True)
                    for p in range(nproc):
                        # Create build folder for each core
                        cmd = f"mkdir -p build_{p}"
                        print(cmd)
                        subprocess.run(cmd, shell=True, check=True)
                        # Generate data
                        cmd = f"./data/data_gen.py -s {s} -n {n} -m"
                        print(cmd)
                        subprocess.run(cmd, shell=True, check=True)
                        # Build
                        cmd = f"cd build_{p} && cmake -DSNITCH_RUNTIME=snRuntime-cluster .. && make -j"
                        print(cmd)
                        subprocess.run(cmd, shell=True, check=True)
                    # Run in parallel
                    processses = []
                    for j in range(nproc):
                        # Make sure to run it in fast mode to prevent recompilation/linking of data
                        cmd = f"cd build_{j} && make run-rtl-{cfg['binary']}/fast"
                        print(cmd)
                        processses.append(subprocess.Popen(cmd, shell=True))
                    # Extract results
                    for j, p in enumerate(processses):
                        p.wait()
                        out_csv = f"{outdir}/{cfg['binary']}_n{n}_s{s}_r{i+j}.csv"
                        # out_csv = f"{outdir}/{cfg['binary']}_n{n}_s{s}_r{i+j}_a{cfg['axis']}.csv"
                        cmd = f"./perf_extr.py -i build_{j}/logs -o {out_csv} -n {n}"
                        print(cmd)
                        subprocess.run(cmd, shell=True, check=True)
                        # Read in results to compute confidence interval
                        df = pd.read_csv(out_csv, index_col=0)
                        cycles = df.loc['mean', 'cycles']
                        logging.info(f"Results of run {i+j}: Cycles: {cycles}")
                        results.append(cycles)
                    # Compute confidence interval
                    m, se, h = mean_confidence_interval(pd.DataFrame(results), 0.95)
                    logging.info(f"Confidence interval: {m} +/- {h}")
                    # Break when confidence interval is smaller than 1% of mean
                    if cfg['num_runs'] > 0:
                        if (h/m < 0.01).bool():
                            break

    # Remove build folders
    cmd = "rm -rf build*"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)





def main():

    script_path = pathlib.Path(__file__).parent.absolute()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run measurements')
    parser.add_argument("-i", "--indir", type=pathlib.Path, default=script_path / "results", required=False,
                        help='Path to input file')
    parser.add_argument("-o", "--outdir", type=pathlib.Path, default=script_path / "results", required=False,
                        help='Path to output directory')
    parser.add_argument("-p", "--plot", action='store_true', help='Plot results')
    parser.add_argument("-r", "--run", action='store_true', help='Run measurements')
    parser.add_argument("-n", "--nproc", type=int, required=False, default=8, help='Number of tests in parallel')
    parser.add_argument("-t", "--num_tests", type=int, required=False, default=50,
                        help='Number of maximum runs, it will stop when confidence interval is small enough')

    args = parser.parse_args()

    size = [8, 16, 32, 64]

    test_cfg = []
    # test_cfg.append({'binary': 'matmul_dense_dense', 'nproc': [1, 8], 'size': size, 'num_runs': 1})
    # test_cfg.append({'binary': 'matmul_csr_dense', 'nproc': [1], 'size': size, 'num_runs': args.num_tests})
    # test_cfg.append({'binary': 'matmul_csr_csr', 'nproc': [1], 'size': size, 'num_runs': args.num_tests})
    # test_cfg.append({'binary': 'matmul_csr_dense_to_dense', 'nproc': [1, 8], 'size': size, 'num_runs': args.num_tests})
    # test_cfg.append({'binary': 'matmul_dense_dense', 'nproc': [8], 'size': size, 'num_runs': 1})
    # test_cfg.append({'binary': 'matmul_csr_dense_to_dense', 'nproc': [8], 'size': size, 'num_runs': args.num_tests})
    test_cfg.append({'binary': 'matmul_csr_csr_to_dense', 'nproc': [8], 'size': size, 'num_runs': args.num_tests})
    #test_cfg.append({'binary': 'softmax_dense', 'nproc': [1, 8], 'size': size, 'num_runs': 1, 'axis': -1})
    #test_cfg.append({'binary': 'softmax_dense', 'nproc': [1, 8], 'size': size, 'num_runs': 1, 'axis': 0})
    #test_cfg.append({'binary': 'softmax_csr', 'nproc': [1, 8], 'size': size, 'num_runs': args.num_tests, 'axis': -1})
    test_cfg.append({'binary': 'softmax_csr', 'nproc': [1], 'size': size, 'num_runs': args.num_tests, 'axis': 0})

    if args.run:
        run_measurements(outdir=args.outdir, test_config=test_cfg, nproc=args.nproc)

    if args.plot:
        plot(indir=args.indir, outdir=args.outdir)


if __name__ == "__main__":
    main()
