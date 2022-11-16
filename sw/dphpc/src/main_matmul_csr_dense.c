// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include <math.h>
#include "data_matmul.h"
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
  res.values = snrt_l3alloc(A.rows * B.cols * sizeof(double));
  res.col_idx = snrt_l3alloc(A.rows * B.cols * sizeof(int));
  res.row_ptr = snrt_l3alloc((A.rows + 1) * sizeof(int));


  // Run the matrix multiplication
  matmul_csr_dense(&A, &B_dense, &res);

  // Check the result
  int errors = 0;
  for (int i = 0; i < res.nnz; i++) {

    if (fabs(res.values[i] - C.values[i]) > 0.001) {
      errors++;
    }
  }

  if (errors != 0) {
    printf("Errors: %d/%d!\n", errors, res.nnz);
  }

  return errors;
}
