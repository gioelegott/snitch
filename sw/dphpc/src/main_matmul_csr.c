// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "data_matmul_csr.h"
#include "matmul_csr.h"
#include "snrt.h"
#include "printf.h"

int main() {

  if (snrt_cluster_core_idx() != 0) return 0;

  csr_matrix res;

  // Initialize the result matrix
  res.rows = A.rows;
  res.cols = B.cols;
  res.nnz = 0;

  // Allocate space for the result matrix
  res.values = snrt_l3alloc(A.rows * B.cols * sizeof(int));
  res.col_idx = snrt_l3alloc(A.rows * B.cols * sizeof(int));
  res.row_ptr = snrt_l3alloc((A.rows + 1) * sizeof(int));

  printf("A.values[0] = %f\n", A.values[0]);
  printf("A.col_idx[0] = %d\n", A.col_idx[0]);


  // Run the matrix multiplication
  matmul_csr(&A, &B, &res);

  // Check the result
  int errors = 0;
  for (int i = 0; i < res.nnz; i++) {
    printf("res.values[%d] = %f\n", i, res.values[i]);
    if (res.values[i] != C.values[i]) {
      errors++;
    }
  }

  if (errors == 0) {
    printf("Test passed!\n");
  }

  return errors

  ;
}
