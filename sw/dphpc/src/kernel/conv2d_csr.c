// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yichao Zhang <yiczhang@iis.ee.ethz.ch>

#include "conv2d_csr.h"
#include "printf.h"

void conv2d_csr(struct csr_matrix **A, struct csr_matrix **filter, struct csr_matrix *res, int channel_in) {

  res->rows = A[0]->rows - filter[0]->rows + 1;
  res->cols = A[0]->cols - filter[0]->cols + 1;

  for (int i = 0; i < res->rows; i++) {
    for (int j = 0; j < res->cols; j++) {
      double sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter[ci]->rows; kx ++) {
          for (int kx_f = filter[ci]->row_ptr[kx]; kx_f < filter[ci]->row_ptr[kx + 1]; kx_f++) {
            for (int kx_a = A[ci]->row_ptr[kx + i]; kx_a < A[ci]->row_ptr[kx + i + 1]; kx_a++)  {
              if (filter[ci]->col_idx[kx_f] + j == A[ci]->col_idx[kx_a]) {
                sum += A[ci]->values[kx_a] * filter[ci]->values[kx_f];
                break;
              }
            }
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
}

void conv2d_dense(struct dense_matrix **A, struct dense_matrix **filter, struct csr_matrix *res, int channel_in) {

  res->rows = A[0]->rows - filter[0]->rows + 1;
  res->cols = A[0]->cols - filter[0]->cols + 1;

  for (int i = 0; i < res->rows; i++) {
    for (int j = 0; j < res->cols; j++) {
      double sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        for (int kx = 0; kx < filter[ci]->rows; kx ++) {
          for (int ky = 0; ky < filter[ci]->cols; ky ++) {
            sum += A[ci]->values[(i * A[ci]->cols)+ j + (kx * A[ci]->cols) + ky] * filter[ci]->values[kx * filter[ci]->cols + ky];
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
}

void conv2d_csr_dense(struct csr_matrix **A, struct dense_matrix **filter, struct csr_matrix *res, int channel_in) {

  res->rows = A[0]->rows - filter[0]->rows + 1;
  res->cols = A[0]->cols - filter[0]->cols + 1;

  for (int i = 0; i < res->rows; i++) {
    for (int j = 0; j < res->cols; j++) {
      double sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter[ci]->rows; kx ++) {
          for (int kx_a = A[ci]->row_ptr[kx + i]; kx_a < A[ci]->row_ptr[kx + i + 1]; kx_a++)  {
            if (A[ci]->col_idx[kx_a] >= j) {
              if (A[ci]->col_idx[kx_a] <= filter[ci]->cols + j -1){
                int f_col = A[ci]->col_idx[kx_a] - j;
                sum += A[ci]->values[kx_a] * filter[ci]->values[kx * filter[ci]->cols + f_col];
              }
            }
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
}
