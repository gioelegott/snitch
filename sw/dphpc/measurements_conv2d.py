#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import logging


import pandas as pd
import numpy as np
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from matplotlib import cm
from matplotlib import colors
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


# Log to measurements.log
logging.basicConfig(filename='measurements.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')


def read_value_from_csv(indir: pathlib.Path, binary: str, nproc: int, size: int, metric: str, stat: str = 'mean'):

    # Read in csv_files
    df = pd.read_csv(indir / (binary + '_'.join([str(p) for p in [nproc, size, 0]]) + '.csv'), index_col=0)
    return df.loc['mean', metric]


def mean_confidence_interval(data, confidence):
    m = data.apply(np.mean, axis=0)
    se = data.apply(stats.sem)
    n = len(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m[0], se[0], h[0]


def run_measurements(outdir: pathlib.Path, test_config: list, nproc: int):

    # Create outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for cfg in test_config:
        for d in cfg['density']:
            for s in cfg['size']:
                for n in cfg['nproc']:
                    results = []
                    if cfg['num_runs'] < nproc:
                        nproc = cfg['num_runs']
                    for i in range(0, cfg['num_runs'], nproc):
                        # Run measurement
                        logging.info(f"Running measurement {i+1}-{i+nproc}/{cfg['num_runs']} of {cfg['binary']} with {d} density, {n} cores and size {s}")
                        
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
                            cmd = f"./data/data_gen.py -m {s} -n {n} -d {d}"
                            print(cmd)
                            subprocess.run(cmd, shell=True, check=True)
                            # Build
                            cmd = f"cd build_{p} && cmake-3.18.1 -DCMAKE_TOOLCHAIN_FILE=toolchain-llvm -DSNITCH_RUNTIME=snRuntime-cluster -DBUILD_TESTS=ON .. && make"
                            print(cmd)
                            subprocess.run(cmd, shell=True, check=True)
                    
                        # Run in parallel
                        processes = []
                        for j in range(nproc):
                            # Make sure to run it in fast mode to prevent recompilation/linking of data
                            cmd = f"cd build_{j} && make run-rtl-{cfg['binary']}/fast"
                            print(cmd)
                            processes.append(subprocess.Popen(cmd, shell=True))
                        
                        # Extract results
                        for j, p in enumerate(processes):
                            p.wait()
                            out_csv = f"{outdir}/{cfg['binary']}_d{d}_n{n}_s{s}_r{i+j}.csv"
                            cmd = f"./perf_extr.py --nproc {n} --section 1 --input ./build_{j}/logs --output {out_csv}"
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
                        if ((i > 10) & (h/m < 0.01)):
                            break
                        if (h/m)==0 :
                            break

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

    test_cfg = []
    density = [0.1]
    size = [4, 8, 16]
    # Small size for fast results
    test_cfg.append({'binary': 'conv2d_csr_csr_dense', 'nproc': [1, 8], 'size': size, 'density': density, 'num_runs': args.num_tests})
    test_cfg.append({'binary': 'conv2d_dense_csr_dense', 'nproc': [1, 8], 'size': size, 'density': density, 'num_runs': args.num_tests})
    test_cfg.append({'binary': 'conv2d_dense_csrr_dense', 'nproc': [1, 8], 'size': size, 'density': density, 'num_runs': args.num_tests})
    test_cfg.append({'binary': 'conv2d_csr_dense_dense', 'nproc': [1, 8], 'size': size, 'density': density, 'num_runs': args.num_tests})
    test_cfg.append({'binary': 'conv2d_dense_dense_dense', 'nproc': [1, 8], 'size': size, 'density': density, 'num_runs': args.num_tests})
    
    size = [32]
    # Large size will take 4 hours for each set
    test_cfg.append({'binary': 'conv2d_csr_csr_dense', 'nproc': [8, 1], 'size': size, 'density': density, 'num_runs': args.num_tests})
    test_cfg.append({'binary': 'conv2d_dense_csr_dense', 'nproc': [8, 1], 'size': size, 'density': density, 'num_runs': args.num_tests})
    test_cfg.append({'binary': 'conv2d_dense_csrr_dense', 'nproc': [1, 8], 'size': size, 'density': density, 'num_runs': args.num_tests})
    test_cfg.append({'binary': 'conv2d_csr_dense_dense', 'nproc': [8, 1], 'size': size, 'density': density, 'num_runs': args.num_tests})
    test_cfg.append({'binary': 'conv2d_dense_dense_dense', 'nproc': [8, 1], 'size': size, 'density': density, 'num_runs': args.num_tests})

    if args.run:
        run_measurements(outdir=args.outdir, test_config=test_cfg, nproc=args.nproc)

    if args.plot:
        plot(indir=args.indir, outdir=args.outdir)


if __name__ == "__main__":
    main()
