// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "softmax_csr.h"
#include "printf.h"

double my_fabs(double x) {
    if(x < 0) {
        return -x;
    } else {
        return x;
    }
}

// INFO: This is a custom function to determine the expponential of a floating point number.
//       We assume here the sum representation of an exponential: exp_n(x) = sum_{i=0}^n (x^i/i!).
//       If two partial sums differ less than epsilon, we can stop the summing.
inline double my_exp(double x) 
{ 
    const double epsilon = 1e-7;
    double sum = 0.0; 
    int n = 0; 
    double factorial = 1; 
    double power = 1.0;
    double term; 
    do {
        term = power / factorial;
        sum += term; 
        n += 1; 
        power *= x; 
        factorial *= n;
    } while (my_fabs(term) >= epsilon);
    return sum; 
}

void softmax_csr(int axis, csr_matrix *A, csr_matrix *res) {

    int i, j, k, m;

    res->rows = A->rows;
    m = A->cols;
    res->cols = m;
    res->nnz = A->rows * A->cols;
    res->row_ptr[0] = 0;

    // For axis zero
    if (axis == 0) {

        k = 0;
        for (i = 0; i < A->rows; i++) {
            // Compute the sum
            double sum = (A->cols) - ((A->row_ptr[i + 1]) - (A->row_ptr[i]));
            for (j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
                sum += my_exp(A->values[j]);
            }
            // Compute the Logits
            for (j = 0; j < A->cols; j++) {
                if ((k < A->row_ptr[i + 1]) && (A->col_idx[k] == j))  {
                    res->values[i * m + j] = my_exp(A->values[k]) / sum;
                    res->col_idx[i * m + j] = j;
                    k++;
                } else {
                    res->values[i * m + j] = 1.0 / sum;
                    res->col_idx[i * m + j] = j;
                }
            }
            res->row_ptr[i + 1] = res->row_ptr[i] + A->cols;
        }

    // For other axes
    } else {

        double logit, nnz_value;

        for (j = 0; j < A->cols; j++) {
            // Compute the sum
            double sum = (double) A->cols;
            for (k = 0; k < A->nnz; k++) {
                if (A->col_idx[k] == j) {
                    nnz_value = A->values[k];
                    sum += (my_exp(nnz_value) - 1);
                }
            }
            // Compute the logits
            for (i = 0; i < A->rows; i++) {
                logit = (double) (1.0 / sum);
                res->values[i * m + j] = logit;
                res->col_idx[i * m + j] = j;
                res->row_ptr[i + 1] += 1;
                k = 0;
                while(k < A->row_ptr[i + 1]) {
                    if (A->col_idx[k] == j) {
                        nnz_value = A->values[k];
                        res->values[i * m + j] = logit * my_exp(nnz_value);
                    }
                    k++;
                }
            }
        }

    }
};
