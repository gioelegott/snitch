// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yichao Zhang <yiczhang@iis.ee.ethz.ch>

#include <math.h>
#include "conv2d_csr.h"
#include "snrt.h"
#include "printf.h"
#include "data_conv2d_csr.h"

int main() {
  if (snrt_cluster_core_idx() != 0) return 0;

  assign_A();
  assign_FILTER();
  assign_RES();
  
  csr_matrix res;  
  // Initialize the result matrix
  res.rows = A[0].rows - FILTER[0].rows +1;
  res.cols = A[0].cols - FILTER[0].cols +1;
  res.nnz = 0;

  // Allocate space for the result matrix
  res.values = snrt_l3alloc(res.rows * res.cols * sizeof(double));
  res.col_idx = snrt_l3alloc(res.rows * res.cols * sizeof(int));
  res.row_ptr = snrt_l3alloc((res.rows + 1) * sizeof(int));

  int input_channels = 2;
  printf("Start Kernel Calculation \n");
  conv2d_csr(A, FILTER, &res, input_channels);
  printf("Finish Kernel Calculation\n");

  // Check the result
  int errors = 0;
  printf("RES has %d non-zero values, RES has %d non-zero values \n", res.nnz, RES.nnz);

  for (int i = 0; i < res.nnz; i++) {
    //printf("res value is %.6f \n", res.values[i]);
    if (fabs(res.values[i] - RES.values[i]) > 0.001) {
      errors++;
    }
  }

  if (errors != 0) {
    printf("Errors: %d/%d!\n", errors, res.nnz);
  }
  
  if (errors == 0) {
    printf("Congratulation! The Results are Correct!\n");
  }

  return errors;
}
