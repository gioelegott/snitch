// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yichao Zhang <yiczhang@iis.ee.ethz.ch>

#include "conv2d_csr.h"
#include "printf.h"

void conv2d_csr_csr_csr(struct csr_matrix **A, struct csr_matrix **filter, struct csr_matrix *res, int channel_in, int filter_row, int res_row, int res_col) {
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j++) {
      double sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
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

void conv2d_csr_csr_dense(struct csr_matrix **A, struct csr_matrix **filter, struct dense_matrix *res, int channel_in, int filter_row, int res_row, int res_col) {
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j++) {
      double sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
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
      res->values[i * res_col + j] = sum;
    }
  }
}

void conv2d_csr_dense_csr(struct csr_matrix **A, struct dense_matrix **filter, struct csr_matrix *res, int channel_in, int filter_row, int filter_col, int res_row, int res_col) {
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j++) {
      double sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
          for (int kx_a = A[ci]->row_ptr[kx + i]; kx_a < A[ci]->row_ptr[kx + i + 1]; kx_a++)  {
            int A_col_idx = A[ci]->col_idx[kx_a];
            if (A_col_idx >= j) {
              if (A_col_idx <= filter_col + j -1){
                int f_idx = kx * filter_col + A[ci]->col_idx[kx_a] - j;
                sum += A[ci]->values[kx_a] * filter[ci]->values[f_idx];
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

void conv2d_csr_dense_dense(struct csr_matrix **A, struct dense_matrix **filter, struct dense_matrix *res, int channel_in, int filter_row, int filter_col, int res_row, int res_col) {
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j++) {
      double sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        // CSR Version Inner Loop:
        for (int kx = 0; kx < filter_row; kx ++) {
          for (int kx_a = A[ci]->row_ptr[kx + i]; kx_a < A[ci]->row_ptr[kx + i + 1]; kx_a++)  {
            int A_col_idx = A[ci]->col_idx[kx_a];
            if (A_col_idx >= j) {
              if (A_col_idx <= filter_col + j -1){
                int f_idx = kx * filter_col + A[ci]->col_idx[kx_a] - j;
                sum += A[ci]->values[kx_a] * filter[ci]->values[f_idx];
              }
            }
          }
        }
      }
      res->values[i * res_col + j] = sum;
    }
  }
}

void conv2d_dense_dense_dense(struct dense_matrix **A, struct dense_matrix **filter, struct dense_matrix *res, int channel_in, int res_row, int res_col) {
  for (int i = 0; i < res_row; i++) {
    for (int j = 0; j < res_col; j++) {
      double sum = 0;
      // "Channel IN" Loop
      for (int ci=0; ci < channel_in; ci++){
        int A_col = A[ci]->cols;
        int filter_col = filter[ci]->cols;
        for (int kx = 0; kx < filter[ci]->rows; kx ++) {
          for (int ky = 0; ky < filter_col; ky ++) {
            sum += A[ci]->values[i * A_col + j + kx * A_col + ky] * filter[ci]->values[kx * filter_col + ky];
          }
        }
      }
      res->values[i * res_col + j] = sum;
    }
  }
}
