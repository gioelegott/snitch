# DPHPC Project: Parallel Implementation of Sparse ONNX Kernels on a Cluster of Lightweight Processors

## Introduction

This folder contains all the source code and the documentation for the DPHPC project. The project is a parallel implementation of sparse ONNX kernels on a cluster of lightweight processors. The project is based on the [Snitch](https://github.com/pulp-platform/snitch) project. We implemented sparse kernels for Softmax, GEMM and Convolution operators. All kernels have been verified and profiled with banshee and with cycle-accurate RTL simulations.

The authors of the project are:
* Tim Fischer `<fischeti@iis.ee.ethz.ch>`
* Marco Bertuletti `<mbertuletti@iis.ee.ethz.ch>`
* Yichao Zhang `<yiczhang@iis.ee.ethz.ch>`

## Contents
* `data`: Data generation scripts and template files for all kernels
   * `data_gen_<kernel>.py`: Python scripts which randomly generate data for the kernels and also calculate the expected output with golden models based on Scipy, Tensorflow etc.
   * `data_<kernel>.h.tpl`: Template files for the data generation scripts. The scripts will fill the template files with the generated data.
   * `data_<kernel>.h`: Generated data files for the kernels.
* `src`: Source code of all SW testbenches and kernels
    * `main_<kernel>.c`: Testbenches for the kernels, each testbench loads the data into L1 before calling the kernel. The testbenches also check the output against the expected output. The performance is also measured with performance counters that are set right before and after calling the corresponding kernel.
    * `kernels`: ONNX kernels for Softmax, GEMM and Convolution
    * `utils`: Some utility functions

## Prerequisites

We recommend to follow [this guide](https://pulp-platform.github.io/snitch/ug/getting_started/) to setup Snitch.

Further, `python3` is required with the packages specified in the `requirements.txt`

## Usage

### Basic usage

To run a binary on Banshee, please type the following commands:

```
mkdir build && cd build
cmake ..
make run-banshee-<kernel>_<formats>
```

where `<kernel>` is one of `softmax`, `matmul` or `conv` and `<formats>` are the corresponding sparse representations.
For example:

```
make run-banshee-matmul_csr_dense
```

To run the kernels on cycle-accurate RTL simulations, you need to compile the simulator first. We recommend to use the openly available verilator simulator. Please refer to [this guide](https://github.com/pulp-platform/snitch/blob/master/hw/system/snitch_cluster/README.md) to compile the verilator executable of a Snitch cluster.

Afterwards, you can run the kernels with the following commands:

```
mkdir build && cd build
cmake -DSNITCH_RUNTIME snRuntime-cluster ..
make run-rtl-<kernel>_<formats>
```

where `<kernel>` and `<formats>` behave as described above.
