#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import logging
import glob

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


def plot(indir: pathlib.Path, outdir: pathlib.Path):

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

    dims = np.array([8, 16, 32, 64])
    dims_normalized = np.log2(dims)*8

    """
    Cycles Plots
    """

    dense_cycles = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=1, size=dims)
    csr_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense', nproc=1, size=dims)
    csr_csr_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_csr', nproc=1, size=dims)
    parallel_dense_cycles = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=dims)
    parallel_csr_barrier_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='synch_overhead')
    parallel_csr_transform_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='transform_overhead')
    parallel_csr_cycles = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims)-parallel_csr_barrier_cycles-parallel_csr_transform_cycles
    norm_csr_cycles = 1/csr_cycles.div(dense_cycles.iloc[0], axis=1)
    norm_csr_csr_cycles = 1/csr_csr_cycles.div(dense_cycles.iloc[0], axis=1)
    norm_dense_cycles = 1/dense_cycles.div(dense_cycles.iloc[0], axis=1)
    norm_parallel_dense_cycles = 1/parallel_dense_cycles.div(dense_cycles.iloc[0], axis=1)
    norm_parallel_csr_cycles = 1/parallel_csr_cycles.div(dense_cycles.iloc[0], axis=1)

    fig0 = plt.figure(figsize=(15, 6))
    ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=1)
    ax1 = plt.subplot2grid((2, 2), (1, 0), rowspan=1)
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    cmap = mcp.gen_color(cmap="Spectral_r",n=6)

    width = 1.5
    m, se, h = mean_confidence_interval(dense_cycles, 0.95)
    l1 = ax0.bar(dims_normalized-width, m, width, color=cmap[0], edgecolor='k')
    m, se, h = mean_confidence_interval(csr_cycles, 0.95)
    l2 = ax0.bar(dims_normalized, m, width, color=cmap[1], edgecolor='k')
    ax0.errorbar(dims_normalized, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m, se, h = mean_confidence_interval(csr_csr_cycles, 0.95)
    l3 = ax0.bar(dims_normalized+width, m, width, color=cmap[2], edgecolor='k')
    ax0.errorbar(dims_normalized+width, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

    ax0.set_xticks(dims_normalized, dims)
    ax0.legend([l1, l2, l3], ['dense', 'CSR-dense', 'CSR-CSR'], loc='upper left', facecolor='white', framealpha=1)
    ax0.set_xlabel('Input dimension')
    ax0.set_ylabel('Cycles')
    ax0.set_title('Matmul Single-core')
    ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax0.grid(True)
    plt.tight_layout()

    width = 1.5
    m1, se, h = mean_confidence_interval(parallel_dense_cycles, 0.95)
    l1 = ax1.bar(dims_normalized-width/2, m1, width, color=cmap[0], edgecolor='k')
    ax1.errorbar(dims_normalized-width/2, m1, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m1, se, h = mean_confidence_interval(parallel_csr_cycles, 0.95)
    l2 = ax1.bar(dims_normalized+width/2, m1, width, color=cmap[1], edgecolor='k')
    ax1.errorbar(dims_normalized+width/2, m1, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m2, se, h = mean_confidence_interval(parallel_csr_barrier_cycles, 0.95)
    l1 = ax1.bar(dims_normalized+width/2, m2, width, color=cmap[0], edgecolor='k', bottom=m1)
    ax1.errorbar(dims_normalized+width/2, m2, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m3, se, h = mean_confidence_interval(parallel_csr_transform_cycles, 0.95)
    l1 = ax1.bar(dims_normalized+width/2, m3, width, color=cmap[0], edgecolor='k', bottom=m1+m2)
    ax1.errorbar(dims_normalized+width/2, m3, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)

    ax1.set_xticks(dims_normalized, dims)
    ax1.legend([l1, l2], ['dense', 'CSR-dense'], loc='upper left', facecolor='white', framealpha=1)
    ax1.set_xlabel('Input dimension')
    ax1.set_ylabel('Cycles')
    ax1.set_title('Matmul Multi-core')
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
    ax1.grid(True)
    plt.tight_layout()

    # m, se, h = mean_confidence_interval(norm_dense_cycles, 0.95)
    # l1 = ax2.bar(dims_normalized-width, m, width, color=cmap[0], edgecolor='k')
    # ax2.errorbar(dims_normalized-width, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m, se, h = mean_confidence_interval(norm_parallel_dense_cycles, 0.95)
    l2 = ax2.bar(dims_normalized-width/2, m, width, color=cmap[1], edgecolor='k')
    ax2.errorbar(dims_normalized-width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    m, se, h = mean_confidence_interval(norm_parallel_dense_cycles, 0.95)
    l3 = ax2.bar(dims_normalized+width/2, m, width, color=cmap[2], edgecolor='k')
    ax2.errorbar(dims_normalized+width/2, m, 2*np.array(h), fmt='none', ecolor='r', elinewidth=2)
    ax2.legend([l2, l3], ['dense', 'CSR-dense'], loc='upper left', facecolor='white', framealpha=1)
    ax2.set_xticks(dims_normalized, dims)
    ax2.set_xlabel('Input dimension')
    ax2.set_ylabel('Speed-up')
    ax2.set_title('Matmul Speed-UP')
    ax2.set(ylim=(0, 10), yticks=np.arange(0, 10, 1))
    ax2.text(42, 8.2, 'IDEAL Speed-UP', color='r', fontsize=MEDIUM_SIZE)
    ax2.axhline(y=8, color='r', linestyle='--')
    ax2.grid(True)
    plt.tight_layout()

    plt.savefig(outdir / 'gemm_cycles.png')
    plt.savefig(outdir / 'gemm_cycles.eps', format='eps')

    """
    IPC Plots
    """

    dense_coreipc = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=1, size=dims, metric='snitch_occupancy')
    dense_fpuipc = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=1, size=dims, metric='fpss_occupancy')
    csr_coreipc = read_run_from_csv(indir=indir, binary='matmul_csr_dense', nproc=1, size=dims, metric='snitch_occupancy')
    csr_fpuipc = read_run_from_csv(indir=indir, binary='matmul_csr_dense', nproc=1, size=dims, metric='fpss_occupancy')
    csr_csr_coreipc = read_run_from_csv(indir=indir, binary='matmul_csr_csr', nproc=1, size=dims, metric='snitch_occupancy')
    csr_csr_fpuipc = read_run_from_csv(indir=indir, binary='matmul_csr_csr', nproc=1, size=dims, metric='fpss_occupancy')
    parallel_dense_coreipc = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=dims, metric='snitch_occupancy')
    parallel_dense_fpuipc = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=dims, metric='fpss_occupancy')
    parallel_dense_synch = read_run_from_csv(indir=indir, binary='matmul_dense_dense', nproc=8, size=dims, metric='synch_overhead')
    parallel_csr_coreipc = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='snitch_occupancy')
    parallel_csr_fpuipc = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='fpss_occupancy')
    parallel_csr_synch = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='synch_overhead')
    parallel_csr_transform = read_run_from_csv(indir=indir, binary='matmul_csr_dense_to_dense', nproc=8, size=dims, metric='transform_overhead')

    fig1 = plt.figure(figsize=(11, 6))
    ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    cmap = mcp.gen_color(cmap="RdBu", n=6)
    cmap = cmap[2:4]
    patterns = ["", "/", ".", "x"]

    ax0.bar(dims_normalized-width, dense_coreipc.apply(np.mean, axis=0), width,
            color=cmap[0], hatch=patterns[0], edgecolor='k')
    ax0.bar(dims_normalized-width, dense_fpuipc.apply(np.mean, axis=0), width,
            bottom=dense_coreipc.apply(np.mean, axis=0),
            color=cmap[1], hatch=patterns[0], edgecolor='k')
    ax0.bar(dims_normalized, csr_coreipc.apply(np.mean, axis=0), width,
            yerr=3*csr_coreipc.apply(np.std, axis=0),
            color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax0.bar(dims_normalized, csr_fpuipc.apply(np.mean, axis=0), width,
            yerr=csr_fpuipc.apply(np.std, axis=0), bottom=csr_coreipc.apply(np.mean, axis=0),
            color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax0.bar(dims_normalized+width, csr_csr_coreipc.apply(np.mean, axis=0), width,
            yerr=3*csr_csr_coreipc.apply(np.std, axis=0),
            color=cmap[0], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax0.bar(dims_normalized+width, csr_csr_fpuipc.apply(np.mean, axis=0), width,
            yerr=3*csr_csr_fpuipc.apply(np.std, axis=0), bottom=csr_csr_coreipc.apply(np.mean, axis=0),
            color=cmap[1], hatch=patterns[2], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

    ax0.legend([plt.bar([0], [0], color=cmap[0], edgecolor='k'),
               plt.bar([0], [0], color=cmap[1], edgecolor='k'),
               plt.bar([0], [0], color='w', edgecolor='k'),
               plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[1]),
               plt.bar([0], [0], color='w', edgecolor='k', hatch=patterns[2])],
               ["INT-core IPC", "FP-SS IPC", "Dense", "CSR-dense", "CSR-CSR"],
               loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='white', framealpha=1)
    ax0.set_xticks(dims_normalized, dims)
    ax0.set(ylim=(0, 1.2), yticks=np.arange(0, 1.3, 0.2))
    ax0.text(42, 1.05, 'IDEAL IPC', color='r', fontsize=MEDIUM_SIZE)
    ax0.axhline(y=1, color='r', linestyle='--')
    ax0.set_xlabel('Input dimension')
    ax0.set_ylabel('Cycles')
    ax0.set_title('Occupation Single-core')
    ax0.grid(True)
    plt.tight_layout()

    s1 = parallel_dense_coreipc.apply(np.mean, axis=0)
    s2 = parallel_dense_fpuipc.apply(np.mean, axis=0)
    s3 = parallel_dense_synch.apply(np.mean, axis=0)
    ax1.bar(dims_normalized-width/2, s1, width, color=cmap[0], hatch=patterns[0], edgecolor='k')
    ax1.bar(dims_normalized-width/2, s2, width, bottom=s1, color=cmap[1], hatch=patterns[0], edgecolor='k')
    ax1.bar(dims_normalized-width/2, s3, width, bottom=s1+s2, color='lightgray', edgecolor='k')

    s1 = parallel_csr_coreipc.apply(np.mean, axis=0)
    s2 = parallel_csr_fpuipc.apply(np.mean, axis=0)
    s3 = parallel_csr_synch.apply(np.mean, axis=0)
    s4 = parallel_csr_transform.apply(np.max, axis=0)
    ax1.bar(dims_normalized+width/2, s1, width, yerr=parallel_csr_coreipc.apply(np.std, axis=0),
            color=cmap[0], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax1.bar(dims_normalized+width/2, s2, width, yerr=parallel_csr_fpuipc.apply(np.std, axis=0), bottom=s1,
            color=cmap[1], hatch=patterns[1], edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax1.bar(dims_normalized+width/2, s3, width, yerr=parallel_csr_synch.apply(np.std, axis=0), bottom=s1+s2,
            color='lightgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))
    ax1.bar(dims_normalized+width/2, s4, width, yerr=parallel_csr_synch.apply(np.std, axis=0), bottom=s1+s2+s3,
            color='darkgray', edgecolor='k', error_kw=dict(ecolor='r', lw=2, capsize=0, capthick=0))

    ax1.legend([plt.bar([3], [0], color=cmap[0], edgecolor='k'),
                plt.bar([3], [0], color=cmap[1], edgecolor='k'),
                plt.bar([3], [0], color='lightgray', edgecolor='k'),
                plt.bar([3], [0], color='w', edgecolor='k'),
                plt.bar([3], [0], color='w', edgecolor='k', hatch=patterns[1])],
               ["INT-core IPC", "FP-SS IPC", "Synch.", "Dense", "CRS"],
               loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='white', framealpha=1)
    ax1.set_xticks(dims_normalized, dims)
    ax1.set(ylim=(0, 1.2), yticks=np.arange(0, 1.3, 0.2))
    ax1.set(xlim=ax0.get_xlim())
    ax1.text(42, 1.05, 'IDEAL IPC', color='r', fontsize=MEDIUM_SIZE)
    ax1.axhline(y=1, color='r', linestyle='--')
    ax1.set_xlabel('Input dimension')
    ax1.set_ylabel('Cycles')
    ax1.set_title('Occupation 8-cores')
    ax1.grid(True)
    plt.tight_layout()

    plt.savefig(outdir / 'gemm_ipc.png')
    plt.savefig(outdir / 'gemm_ipc.eps', format='eps')


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
    test_cfg.append({'binary': 'matmul_csr_dense_to_dense', 'nproc': [8], 'size': size, 'num_runs': args.num_tests})

    if args.run:
        run_measurements(outdir=args.outdir, test_config=test_cfg, nproc=args.nproc)

    if args.plot:
        plot(indir=args.indir, outdir=args.outdir)


if __name__ == "__main__":
    main()
