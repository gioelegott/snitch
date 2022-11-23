// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "snrt.h"
#include "utils.h"
#include "printf.h"

#include "data_softmax_csr.h"
#include "softmax_csr.h"

#define SINGLE
#define AXIS (1)

csr_matrix volatile res;
csr_matrix volatile in;

int volatile errors;
double volatile ERROR = 1e-3;

int main() {

  int core_id = snrt_cluster_core_idx();
  int nPE = snrt_cluster_core_num();

#if defined(SINGLE)

  if (core_id != 0) return 0;

  // Allocate space for the result matrix
  res.values = snrt_l1alloc(2 * A.rows * A.cols * sizeof(int));
  res.col_idx = snrt_l1alloc(A.rows * A.cols * sizeof(int));
  res.row_ptr = snrt_l1alloc((A.rows + 1) * sizeof(int));
  // Allocate space for the input matrix
  in.values = snrt_l1alloc(2 * A.nnz * sizeof(int));
  in.col_idx = snrt_l1alloc(A.nnz * sizeof(int));
  in.row_ptr = snrt_l1alloc((A.rows + 1) * sizeof(int));

  // Initialize the result matrix
  res.rows = A.rows;
  res.cols = A.cols;
  res.nnz = A.rows * A.cols;
  // Initialize the input matrix
  in.rows = A.rows;
  in.cols = A.cols;
  in.nnz = A.nnz;
  for(int i = 0; i < A.nnz; i++) {
    in.values[i] = A.values[i];
    in.col_idx[i] = A.col_idx[i];
  }
  for(int i = 0; i < (A.rows + 1); i++) {
    in.row_ptr[i] = A.row_ptr[i];
  }
  printf("A.values[0] = %f\n", A.values[0]);

  // Run the softmax
  softmax_csr_single(AXIS, &in, res.values);
  size_t time_init = benchmark_get_cycle();
  softmax_csr_single(AXIS, &in, res.values);
  size_t time_end = benchmark_get_cycle();

  // Check the result
  for (int i = 0; i < res.nnz; i++) {
    // printf("res.values[%d] = %f\n", i, res.values[i]);
    if (my_fabs(res.values[i] - C.values[i]) > ERROR) {
      errors++;
    }
  }
  if (errors == 0) {
    printf("Test passed!\n");
  }

  return errors;

#elif defined(PARALLEL)

  if (core_id == 0) {
    // Allocate space for the result matrix
    res.values = snrt_l1alloc(2 * A.rows * A.cols * sizeof(int));
    res.col_idx = snrt_l1alloc(A.rows * A.cols * sizeof(int));
    res.row_ptr = snrt_l1alloc((A.rows + 1) * sizeof(int));
    // Allocate space for the input matrix
    in.values = snrt_l1alloc(2 * A.nnz * sizeof(int));
    in.col_idx = snrt_l1alloc(A.nnz * sizeof(int));
    in.row_ptr = snrt_l1alloc((A.rows + 1) * sizeof(int));

    // Initialize the result matrix
    res.rows = A.rows;
    res.cols = A.cols;
    res.nnz = A.rows * A.cols;
    // Initialize the input matrix
    in.rows = A.rows;
    in.cols = A.cols;
    in.nnz = A.nnz;
    for(int i = 0; i < A.nnz; i++) {
        in.values[i] = A.values[i];
        in.col_idx[i] = A.col_idx[i];
    }
    for(int i = 0; i < (A.rows + 1); i++) {
        in.row_ptr[i] = A.row_ptr[i];
    }
  }
  snrt_cluster_hw_barrier();

  // Run the softmax
  if (core_id != 8) {
    softmax_csr_parallel(AXIS, &in, res.values, core_id, nPE);
    size_t time_init = benchmark_get_cycle();
    softmax_csr_parallel(AXIS, &in, res.values, core_id, nPE);
    size_t time_end = benchmark_get_cycle();
  }
  snrt_cluster_hw_barrier();

  if (core_id == 0) {
    // Check the result
    errors = 0;
    for (int i = 0; i < res.nnz; i++) {
        if (my_fabs(res.values[i] - C.values[i]) > ERROR) {
            errors++;
        }
    }
    if (errors == 0) {
        printf("Test passed!\n");
    }
  }
  snrt_cluster_hw_barrier();

  return errors;

#endif

}
