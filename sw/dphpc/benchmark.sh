#!/bin/bash
# Benchmark shell script

DPHPC_DIR=$(dirname $(realpath “${BASH_SOURCE:-$0}”))
BUILD_DIR="$DPHPC_DIR/build"
RESULTS_DIR="$DPHPC_DIR/results/conv2d_benchmark"
MODE="sparse_parallel"

echo $BUILD_DIR
echo $RESULTS_DIR
echo "doing $MODE mode benchmarking"

cd "$BUILD_DIR"

cmake-3.18.1 -DCMAKE_TOOLCHAIN_FILE=toolchain-llvm -DSNITCH_RUNTIME=snRuntime-cluster -DBUILD_TESTS=ON ..
make

for channel in 8
do
    for size in 4 8 16 32
    do
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "----------------------------------------------------"
        echo "Iteration on SPARSE ${channel}ch_${size}x${size}_3x3_0.1den"
        echo "Results will be collected in path: $RESULTS_DIR/${MODE}_${channel}ch_${size}x${size}_3x3_0.1den"
        echo "----------------------------------------------------"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

        cd "$BUILD_DIR/../data"
        python3 data_gen.py --channel_size $channel --matrix_size $size --filter_size 3
        cd $BUILD_DIR
        make run-rtl-conv2d_csr > ${BUILD_DIR}/logs/run.log
        cp -rf "$BUILD_DIR/logs" "$RESULTS_DIR/${MODE}_${channel}ch_${size}x${size}_3x3_0.1den"
        mv "$BUILD_DIR/conv2d_csr_perf.csv" "$RESULTS_DIR/${MODE}_${channel}ch_${size}x${size}_3x3_0.1den/."
        mv "$BUILD_DIR/conv2d_csr.s" "$RESULTS_DIR/${MODE}_${channel}ch_${size}x${size}_3x3_0.4den/."
    done
done

MODE="dense_parallel"
echo "doing $MODE mode benchmarking"

for channel in 8
do
    for size in 4 8 16 32
    do
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "----------------------------------------------------"
        echo "Iteration on DENSE ${channel}ch_${size}x${size}_3x3_0.1den"
        echo "Results will be collected in path: $RESULTS_DIR/${MODE}_${channel}ch_${size}x${size}_3x3_0.1den"
        echo "----------------------------------------------------"
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

        cd "$BUILD_DIR/../data"
        python3 data_dense_gen.py --channel_size $channel --matrix_size $size --filter_size 3
        cd $BUILD_DIR
        make run-rtl-conv2d_dense > ${BUILD_DIR}/logs/run.log
        cp -rf "$BUILD_DIR/logs" "$RESULTS_DIR/${MODE}_${channel}ch_${size}x${size}_3x3_0.1den"
        mv "$BUILD_DIR/conv2d_dense_perf.csv" "$RESULTS_DIR/${MODE}_${channel}ch_${size}x${size}_3x3_0.1den/."
        mv "$BUILD_DIR/conv2d_dense.s" "$RESULTS_DIR/${MODE}_${channel}ch_${size}x${size}_3x3_0.4den/."
    done
done





