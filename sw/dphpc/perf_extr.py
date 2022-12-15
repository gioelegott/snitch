#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

import pandas as pd
import argparse
import pathlib


def extract_perf_data(log_file: str, proc_id: int) -> list:

    # Read log file
    with open(log_file, "r") as f:
        extracting = False
        metrics_list = []
        metrics = {}
        # Read linse until Performance metrics are found
        for line in f:
            if line.startswith("Performance metrics"):
                extracting = True
                metrics = {"desc": f"proc_{proc_id}"}
                continue
            # If line is empty or has newline, stop extracting
            if line == "\n" and extracting:
                extracting = False
                metrics_list.append(metrics)
                metrics = {}
                continue
            if extracting:
                # extract first word of line
                metric_key = line.split()[0]
                # detect if last metric is a float or a hex value
                try:
                    metric_value = float(line.split()[-1])
                except ValueError:
                    metric_value = int(line.split()[-1], 16)
                # extract number from string
                metrics[metric_key] = metric_value

        # Append metrics to list
        if metrics != {}:
            metrics_list.append(metrics)
        return metrics_list


def main():

    script_path = pathlib.Path(__file__).parent.absolute()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract performance data from log files')
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        default=script_path / "build/logs",
        required=False,
        help='Path to log files'
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=script_path / "perf_data.csv",
        required=False,
        help='Path to output file'
    )
    parser.add_argument(
        "-s",
        "--section",
        type=int,
        default=1,
        required=False,
        help='Section to extract data from'
    )
    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        default=8,
        required=False,
        help='Number of processes'
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help='Set verbose'
    )

    args = parser.parse_args()

    # Get all n first log files
    log_files = sorted(args.input.glob("trace_*.txt"))[:args.nproc]

    metrics_list = []

    # Extract performance data
    for p, log_file in enumerate(log_files[:args.nproc]):
        # Extract performance data of current core file
        metrics_of_current_proc = extract_perf_data(log_file, p)
        # Append specific section to list of all cores
        try:
            metrics_list.append(metrics_of_current_proc[args.section])
        except IndexError:
            # print with orange color
            print(f"\033[33mWarning: Section {args.section} not found in {log_file}\033[0m")

    # Convert list of dict to pandas dataframe
    df = pd.DataFrame(metrics_list)

    # compute metrics
    df_mean = df.mean(numeric_only=True)
    df_std = df.std(numeric_only=True)
    df_min = df.min()
    df_max = df.max()
    df_total = df.sum()
    # append mean and std to dataframe
    # concat as a new row
    df = pd.concat([df,
                    df_mean.to_frame().T,
                    df_std.to_frame().T,
                    df_min.to_frame().T,
                    df_max.to_frame().T,
                    df_total.to_frame().T])
    # rename last two rows
    df.iloc[-2, 0] = "mean"
    df.iloc[-1, 0] = "std"
    df.iloc[-3, 0] = "min"
    df.iloc[-4, 0] = "max"
    df.iloc[-5, 0] = "total"
    # pretty print dataframe without index
    print(df)

    # write dataframe to csv file
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()