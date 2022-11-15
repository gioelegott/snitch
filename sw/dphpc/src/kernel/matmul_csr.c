// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "matmul_csr.h"
#include "printf.h"

void matmul_csr(csr_matrix *A, csr_matrix *B, csr_matrix *res) {

  res->rows = A->rows;
  res->cols = B->cols;

  for (int i = 0; i < A->rows; i++) {
    for (int j = 0; j < B->cols; j++) {
      double sum = 0;
      for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
        for (int l = B->row_ptr[A->col_idx[k]]; l < B->row_ptr[A->col_idx[k] + 1]; l++) {
          if (B->col_idx[l] == j) {
            sum += A->values[k] * B->values[l];
            break;
          }
        }
      }
      if (sum != 0.0) {
        res->values[res->nnz] = sum;
        res->col_idx[res->nnz] = j;
        res->nnz++;
      }
    }
    res->row_ptr[i + 1] = res->nnz;
  }
};
