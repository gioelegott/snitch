#!/bin/bash
# Benchmark shell script

axis=0

HW_DIR=$(dirname $(realpath “${BASH_SOURCE:-$0}”))
SW_DIR="$HW_DIR/../../../sw/dphpc"
RT_DIR="$HW_DIR/../../.."
RESULTS_DIR="$RT_DIR/results_softmaxdense_sc_axis0"
echo $HW_DIR
echo $SW_DIR
echo $RT_DIR

for dim in 5 10 20 30 40 50
do
mkdir "$RESULTS_DIR/dim_$dim"
    for i in 1 2 3 4 5 6 7 8 9 10
    do
        echo "Iteration $i"
        cd "$SW_DIR/data"
        python3 data_gen_softmax_dense.py --dimension $dim --axis $axis
        cd $HW_DIR
        make softmax_dense -C sw/build/dphpc
        ./bin/snitch_cluster.vlt ./sw/build/dphpc/softmax_dense
        make traces
        mv "$HW_DIR/logs" "$RESULTS_DIR/dim_$dim/logs$i"
        cd $SW_DIR
        echo "$RESULTS_DIR/dim_$dim/logs$i/results.csv"
        ./perf_extr.py --nproc 8 --section 1 --input "$RESULTS_DIR/dim_$dim/logs$i" --output "$RESULTS_DIR/dim_$dim/logs$i/results.csv"
    done
done
