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

  int channels = 8;

  assign_A();
  assign_FILTER();
  assign_RES();
  
  csr_matrix res[channels];  
  for (int i = 0; i < channels; i++) {
    // Initialize the result matrix
    res[i].rows = A[0].rows - FILTER[i][0].rows +1;
    res[i].cols = A[0].cols - FILTER[i][0].cols +1;
    res[i].nnz = 0;
    // Allocate space for the result matrix
    res[i].values = snrt_l3alloc(res[i].rows * res[i].cols * sizeof(double));
    res[i].col_idx = snrt_l3alloc(res[i].rows * res[i].cols * sizeof(int));
    res[i].row_ptr = snrt_l3alloc((res[i].rows + 1) * sizeof(int));
  }

  printf("Start Kernel Calculation \n");
  for (int i = 0; i < channels; i++) { 
    conv2d_csr(A, FILTER[i], &res[i], channels);
  }
  printf("Finish Kernel Calculation\n");

  // Check the result
  int errors = 0;
  
  for (int i = 0; i < channels; i++) {
    printf("res[%d] has %d non-zero values, RES[%d] has %d non-zero values \n", i, res[i].nnz, i, RES[i].nnz);
    for (int j = 0; j < res[i].nnz; j++) {
      if (fabs(res[i].values[j] - RES[i].values[j]) > 0.001) {
        errors++;
      }
    }
    if (errors != 0) {
      printf("Errors: %d/%d!\n", errors, res[i].nnz);
    }
  }

  
  if (errors == 0) {
    printf("Congratulation! The Results are Correct!\n");
  }

  return errors;
}
