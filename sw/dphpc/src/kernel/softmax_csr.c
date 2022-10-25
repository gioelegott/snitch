// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "softmax_csr.h"
#include "printf.h"

// INFO: This is a custom function to determine the expponential of a floating point number.
//       We assume here the sum representation of an exponential: exp_n(x) = sum_{i=0}^n (x^i/i!).
//       If two partial sums differ less than epsilon, we can stop the summing.
inline double my_exp(double x) 
{ 
    const double epsilon = 1e-7; 
    double sum = 0.0; 
    int n = 0; 
    double factorial = 1; 
    double power=1.0; 
    double term; 
    do { 
        term = power/factorial; 
        sum += term; 
        n += 1; 
        power *= x; 
        factorial *=n; 
    } while (my_fabs(term)>=epsilon); 
    return sum; 
} 

void softmax_csr(int axis, csr_matrix *A, csr_matrix *res) {

    printf("A->col_idx[0] = %d\n", A->col_idx[0]);

    res->rows = A->rows;
    res->cols = A->cols;
    res->nnz = A->nnz;

    // For axis zero
    if (axis == 0) {

        for (int i = 0; i < A->rows; i++) {
            // Compute the sum
            double sum = 0;
            for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
                sum += my_exp(A->values[k]);
            }
            // Compute the Logits
            for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
                res->values[k] = my_exp(A->values[k])/sum;
                res->col_idx[k] = A->col_idx[k];
            }
            res->row_ptr[i] = A->row_ptr[i];
        }

    // For other axes
    } else {

        double sum[A->cols];
        for (int i = 0; i < A->cols; i++) {
            sum = 0;
        }
        // Compute the sum
        for (int i = 0; i < A->rows; i++) {
            for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
                sum[A->col_idx[k]] += my_exp(A->values[k]);
                res->col_idx[k] = A->col_idx[k];
                res->values[k] = my_exp(A->values[k]);
            }
            res->row_ptr[i] = A->row_ptr[i];
        }
        // Compute the Logits
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
            for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
                res->values[k] /= sum[res->col_idx[k]];
            }
        }

    }
};
