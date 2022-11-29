// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yichao Zhang <yiczhang@iis.ee.ethz.ch>

///////////////////////////////////////////////////////////////////////
//////////////////////////      HEAD        ///////////////////////////
///////////////////////////////////////////////////////////////////////
#include <math.h>
#include "conv2d_csr.h"
#include "utils.h"
#include "snrt.h"
#include "printf.h"
#include "data_conv2d_csr.h"

///////////////////////////////////////////////////////////////////////
//////////////////////////     CONFIG       ///////////////////////////
///////////////////////////////////////////////////////////////////////
#define NUM_COMP_CORES 1
#define CHANNELS 2
// Declare output matrix
csr_matrix *matrix_A[CHANNELS], *matrix_FILTER[CHANNELS][CHANNELS], *matrix_res[CHANNELS];  

///////////////////////////////////////////////////////////////////////
//////////////////////////      MAIN        ///////////////////////////
///////////////////////////////////////////////////////////////////////
int main() {
  
  // ``````````````````````````//
  //####### Matrix Init #######//
  // ......................... //
  assign_A();
  assign_FILTER();
  assign_RES();
  snrt_cluster_hw_barrier();
  // ``````````````````````````//
  //####### Matrix Alloc ######//
  // ......................... //
  if (snrt_is_dm_core()) {
    for (int j = 0; j < CHANNELS; j++) {

      // Allocate memory for matrix data A
      matrix_A[j] = snrt_l1alloc(sizeof(csr_matrix));
      matrix_A[j]->values = snrt_l1alloc(A[j].nnz * sizeof(double));
      matrix_A[j]->col_idx = snrt_l1alloc(A[j].nnz  * sizeof(int));
      matrix_A[j]->row_ptr = snrt_l1alloc((A[j].rows+1) * sizeof(int));
      matrix_A[j]->nnz = A[j].nnz;
      matrix_A[j]->rows = A[j].rows;
      matrix_A[j]->cols = A[j].cols;

      // Allocate memory for filter data FILTER
      for (int k = 0; k < CHANNELS; k++) {
        matrix_FILTER[k][j] = snrt_l1alloc(sizeof(csr_matrix));
        matrix_FILTER[k][j]->values = snrt_l1alloc(FILTER[k][j].nnz * sizeof(double));
        matrix_FILTER[k][j]->col_idx = snrt_l1alloc(FILTER[k][j].nnz  * sizeof(int));
        matrix_FILTER[k][j]->row_ptr = snrt_l1alloc((FILTER[k][j].rows+1) * sizeof(int));
        matrix_FILTER[k][j]->nnz = FILTER[k][j].nnz;
        matrix_FILTER[k][j]->rows = FILTER[k][j].rows;
        matrix_FILTER[k][j]->cols = FILTER[k][j].cols;
      }

      // Allocate memory for matrix data res
      matrix_res[j] = snrt_l1alloc(sizeof(csr_matrix));
      matrix_res[j]->values = snrt_l1alloc(RES[j].nnz * sizeof(double));
      matrix_res[j]->col_idx = snrt_l1alloc(RES[j].nnz  * sizeof(int));
      matrix_res[j]->row_ptr = snrt_l1alloc((RES[j].rows+1) * sizeof(int));

      // Copy matrix data to L1
      snrt_dma_start_1d((void *)matrix_A[j]->values, (void *)A[j].values, A[j].nnz * sizeof(double));
      snrt_dma_start_1d((void *)matrix_A[j]->col_idx, A[j].col_idx, A[j].nnz  * sizeof(int));
      snrt_dma_start_1d((void *)matrix_A[j]->row_ptr, (void *)A[j].row_ptr, (A[j].rows+1) * sizeof(int));
      for (int k = 0; k < CHANNELS; k++) {
        snrt_dma_start_1d((void *)matrix_FILTER[k][j]->values, (void *)FILTER[k][j].values, FILTER[k][j].nnz * sizeof(double));
        snrt_dma_start_1d((void *)matrix_FILTER[k][j]->col_idx, (void *)FILTER[k][j].col_idx, FILTER[k][j].nnz  * sizeof(int));
        snrt_dma_start_1d((void *)matrix_FILTER[k][j]->row_ptr, (void *)FILTER[k][j].row_ptr, (FILTER[k][j].rows+1) * sizeof(int));
      }
    }
    // Wait for DMA to finish
    snrt_dma_wait_all();
  }
  // Wait for all cores to finish DMA
  snrt_cluster_hw_barrier();

  // ``````````````````````````//
  //####### Calcul Start ######//
  // ......................... //
  int errors = 0;
  
  if (snrt_cluster_core_idx() == 0) {
    // Calculation
    printf("Start Kernel Calculation \n");
    benchmark_get_cycle();
    for (int i = 0; i < CHANNELS; i++) { 
      conv2d_csr(matrix_A, matrix_FILTER[i], matrix_res[i], CHANNELS);
    }
    benchmark_get_cycle();
    printf("Finish Kernel Calculation\n");

    // Check the result
    for (int i = 0; i < CHANNELS; i++) {
      //printf("matrix_res[%d] has %d non-zero values, RES[%d] has %d non-zero values \n", i, matrix_res[i]->nnz, i, RES[i].nnz);
      for (int j = 0; j < matrix_res[i]->nnz; j++) {
        if (fabs(matrix_res[i]->values[j] - RES[i].values[j]) > 0.001) {
          errors++;
        }
      }
      if (errors != 0) {
        printf("Errors: %d/%d!\n", errors, matrix_res[i]->nnz);
      }
    }
    if (errors == 0) {
      printf("Congratulation! The Results are Correct!\n");
    }
  
  }
  
  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  
  return errors;
}
