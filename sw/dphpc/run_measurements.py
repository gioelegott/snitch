#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from matplotlib import cm
from matplotlib import colors
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def read_value_from_csv(indir: pathlib.Path, binary: str, nproc: int, size: int, metric: str, stat: str = 'mean'):

        # Read in csv_files
        df = pd.read_csv(indir / (binary + '_'.join([str(p) for p in [nproc, size, 0]]) + '.csv'), index_col=0)
        return df.loc['mean', metric]


def run_measurements(outdir: pathlib.Path, binary: pathlib.Path, size: list, nproc: list, num_runs: int):

    # Create outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for n in nproc:
        for s in size:
            for i in range(num_runs):
                # Run measurement
                print(f"Running measurement {i+1}/{num_runs}")
                # Clean
                cmd = "cd build && cmake -DSNITCH_RUNTIME=snRuntime-cluster .. && make clean && cd .."
                print(cmd)
                subprocess.run(cmd, shell=True, check=True)
                # Generate data
                cmd = f"./data/data_gen.py -s {s} -n {n} -m"
                print(cmd)
                subprocess.run(cmd, shell=True, check=True)
                # Build & Run
                cmd = f"cd build && make run-rtl-{binary} && cd .."
                print(cmd)
                subprocess.run(cmd, shell=True, check=True)
                # Extract results
                cmd = f"./perf_extr.py -i build/logs -o {outdir / (binary.stem + '_'.join([str(p) for p in [n, s, i]]) + '.csv')} -n {n}"
                print(cmd)
                subprocess.run(cmd, shell=True, check=True)


def plot(indir: pathlib.Path, outdir: pathlib.Path, size: list, nproc: list, num_runs: int):

    SMALL_SIZE = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22

    mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
    mpl.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    mpl.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    mpl.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the x and y labels
    mpl.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    mpl.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    mpl.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fe = font_manager.FontEntry(
        fname='/home/mbertuletti/.local/share/fonts/helvetica/Helvetica.otf',
        name='FreeSans'
    )
    font_manager.fontManager.ttflist.insert(0, fe)
    mpl.rcParams['font.family'] = fe.name
    cmap = mcp.gen_color(cmap="Spectral_r",n=6)

    dims = np.array([8, 16, 32])
    dims_normalized = np.log2(dims)*8

    dense_results = []
    csr_results = []
    csr_csr_results = []
    norm_dense_results = []
    norm_csr_results = []
    norm_csr_csr_results = []
    parallel_dense_results = []
    parallel_csr_results = []
    norm_parallel_dense_results = []
    norm_parallel_csr_results = []
    for i in dims:
        dense_results.append(read_value_from_csv(indir=indir, binary='matmul_dense_dense', nproc=1, size=i,
                             metric='cycles', stat='mean'))
        csr_results.append(read_value_from_csv(indir=indir, binary='matmul_csr_dense', nproc=1, size=i,
                           metric='cycles', stat='mean'))
        csr_csr_results.append(read_value_from_csv(indir=indir, binary='matmul_csr_csr', nproc=1, size=i,
                               metric='cycles', stat='mean'))
        norm_csr_results.append(dense_results[-1]/csr_results[-1])
        norm_csr_csr_results.append(dense_results[-1]/csr_csr_results[-1])
        norm_dense_results.append(dense_results[-1]/dense_results[-1])
        parallel_dense_results.append(read_value_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=i,
                                      metric='cycles', stat='mean'))
        parallel_csr_results.append(read_value_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=i,
                                    metric='cycles', stat='mean'))
        norm_parallel_dense_results.append(dense_results[-1]/parallel_dense_results[-1])
        norm_parallel_csr_results.append(dense_results[-1]/parallel_csr_results[-1])

    fig0, ax = plt.subplots()
    width = 2
    l1 = ax.bar(dims_normalized-width, dense_results, width, color=cmap[0], edgecolor='k')
    l2 = ax.bar(dims_normalized, csr_results, width, color=cmap[1], edgecolor='k')
    l3 = ax.bar(dims_normalized+width, csr_csr_results, width, color=cmap[2], edgecolor='k')
    ax.set_xticks(dims_normalized, dims)
    ax.legend([l1, l2, l3], ['dense', 'CSR-dense', 'CSR-CSR'], loc='upper left', facecolor='white', framealpha=1)
    ax.set_xlabel('Input dimension')
    ax.set_ylabel('Cycles')
    ax.set_title('Matmul Single-core')
    ax.set(ylim=(0, 300000), yticks=np.arange(0, 300000, 50000))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
    ax.grid(True)
    plt.savefig(outdir / 'gemm_kernel_comparison.png')

    norm_dense_results = []
    norm_csr_results = []
    norm_csr_csr_results = []
    for i in range(len(dense_results)):
        norm_csr_results[i] = dense_results[i]/csr_results[i]
        norm_csr_csr_results[i] = dense_results[i]/csr_csr_results[i]
        norm_dense_results[i] = dense_results[i]/dense_results[i]

    fig0, ax = plt.subplots()
    l1 = ax.bar(dims_normalized-width, norm_dense_results, width, color=cmap[0], edgecolor='k')
    l2 = ax.bar(dims_normalized, norm_csr_results, width, color=cmap[1], edgecolor='k')
    l3 = ax.bar(dims_normalized+width, norm_csr_csr_results, width, color=cmap[2], edgecolor='k')
    ax.set_xticks(dims_normalized, dims)
    ax.legend([l1, l2, l3], ['dense', 'CSR-dense', 'CSR-CSR'], loc='upper left', facecolor='white', framealpha=1)
    ax.set_xlabel('Input dimension')
    ax.set_ylabel('Speedup')
    ax.set_title('Matmul Single-core Speedup')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
    ax.grid(True)
    plt.savefig(outdir / 'gemm_kernel_speed_up.png')

    fig0, ax = plt.subplots()
    l1 = ax.bar(dims_normalized-width, dense_results, width, color=cmap[0], edgecolor='k')
    l2 = ax.bar(dims_normalized, csr_results, width, color=cmap[1], edgecolor='k')
    l3 = ax.bar(dims_normalized+width, csr_csr_results, width, color=cmap[2], edgecolor='k')
    ax.set_xticks(dims_normalized, dims)
    ax.legend([l1, l2, l3], ['dense', 'CSR-dense', 'CSR-CSR'], loc='upper left', facecolor='white', framealpha=1)
    ax.set_xlabel('Input dimension')
    ax.set_ylabel('Speedup')
    ax.set_title('Matmul Single-core Speedup')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
    ax.grid(True)
    plt.savefig(outdir / 'gemm_kernel_speed_up.png')


def main():

    script_path = pathlib.Path(__file__).parent.absolute()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run measurements')
    parser.add_argument(
        "-i",
        "--indir",
        type=pathlib.Path,
        default=script_path / "results",
        required=False,
        help='Path to input file'
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=pathlib.Path,
        default=script_path / "results",
        required=False,
        help='Path to output directory'
    )
    parser.add_argument(
        "-p",
        "--plot",
        action='store_true',
        help='Plot results'
    )
    parser.add_argument(
        "-r",
        "--run",
        action='store_true',
        help='Run measurements'
    )
    parser.add_argument(
        "-b",
        "--binary",
        type=pathlib.Path,
        required=False,
        help='Path to binary'
    )
    parser.add_argument(
        "-n",
        "--num_runs",
        type=int,
        required=False,
        default=1,
        help='Number of runs'
    )

    args = parser.parse_args()

    size = [2**i for i in range(3, 7)]
    nproc = [1]

    if args.run:
        run_measurements(outdir=args.outdir,
                         binary=args.binary,
                         num_runs=args.num_runs,
                         size=size,
                         nproc=nproc)

    if args.plot:
        plot(indir=args.indir,
             outdir=args.outdir,
             num_runs=args.num_runs,
             size=size,
             nproc=nproc)


if __name__ == "__main__":
    main()
