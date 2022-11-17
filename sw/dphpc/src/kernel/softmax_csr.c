// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Tim Fischer <fischeti@iis.ee.ethz.ch>

#include "softmax_csr.h"
#include "snrt.h"
#include "printf.h"

#define ABS(x) ((x < 0) ? (-x) : (x))

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
//    const double epsilon = 1e-3;
//    double sum = 0.0;
//    int n = 0;
//    double factorial = 1;
//    double power = 1.0;
//    double term;
//    do {
//        term = power * factorial;
//        sum += term;
//        n += 1;
//        power *= x;
//        factorial /= n;
//    } while (ABS(term) >= epsilon);

//    double volatile sum = 1.0 + x;
//    double volatile factorial_2 = 0.5000000000;
//    double volatile factorial_3 = 0.1666666667;
//    double volatile factorial_4 = 0.0416666667;
//    double volatile factorial_5 = 0.0083333333;
//    double volatile factorial_6 = 0.0013888889;
//    double power_2 = x * x;
//    double power_3 = x * power_2;
//    double power_4 = x * power_3;
//    double power_5 = x * power_4;
//    double power_6 = x * power_5;
//    sum += power_2 * factorial_2;
//    sum += power_3 * factorial_3;
//    sum += power_4 * factorial_4;
//    sum += power_5 * factorial_5;
//    sum += power_6 * factorial_6;
    double volatile sum = 1.0 + x;
    double volatile factorial_2 = 0.5000000000;
    double volatile factorial_3 = 0.1666666667;
    double volatile factorial_4 = 0.0416666667;
    double volatile factorial_5 = 0.0083333333;
    double volatile factorial_6 = 0.0013888889;
    double power_2, power_3, power_4, power_5, power_6;
    asm volatile(
        "fmul.d %[power_2], %[x], %[x];"
        "fmul.d %[power_3], %[power_2], %[x];"
        "fmul.d %[power_4], %[power_2], %[power_2];"
        "fmul.d %[power_5], %[power_2], %[power_3];"
        "fmul.d %[power_6], %[power_3], %[power_3];"
        "fmul.d %[power_2], %[power_2], %[factorial_2];"
        "fmul.d %[power_3], %[power_3], %[factorial_3];"
        "fmul.d %[power_4], %[power_4], %[factorial_4];"
        "fadd.d %[sum], %[power_2], %[sum];"
        "fmul.d %[power_5], %[power_5], %[factorial_5];"
        "fadd.d %[sum], %[power_3], %[sum];"
        "fmul.d %[power_6], %[power_6], %[factorial_6];"
        "fadd.d %[sum], %[power_4], %[sum];"
        "fadd.d %[sum], %[power_5], %[sum];"
        "fadd.d %[sum], %[power_6], %[sum];"
        : [power_2] "+&f" (power_2), [power_3] "+&f" (power_3),
          [power_4] "+&f" (power_4), [power_5] "+&f" (power_5), [power_6] "+&f" (power_6), [sum] "+&f" (sum)
        : [factorial_2] "f" (factorial_2), [factorial_3] "f" (factorial_3),
          [factorial_4] "f" (factorial_4), [factorial_5] "f" (factorial_5), [factorial_6] "f" (factorial_6),
          [x] "f" (x)
        :
    );
    return sum; 
}

void softmax_csr_single(int axis, csr_matrix volatile *A, csr_matrix volatile *res) {

    int i, j, k;
    int n_rows, n_cols, n_nnz;
    int elr, elr_next;
    double logit, nnz_value;
    double sum;

    n_rows = A->rows;
    n_cols = A->cols;
    n_nnz = A->nnz;

    res->rows = n_rows;
    res->cols = n_cols;
    res->nnz = n_rows * n_cols;
    res->row_ptr[0] = 0;

    // For axis zero
    if (axis == 0) {

        k = 0;
        for (i = 0; i < n_rows; i++) {
            elr = A->row_ptr[i];
            elr_next = A->row_ptr[i + 1];
            // Compute the sum
            sum = n_cols - (elr_next - elr);
            for (j = elr; j < elr_next; j++) {
                sum += my_exp(A->values[j]);
            }
            // Compute the Logits
            logit = (double) (1.0 / sum);
            for (j = 0; j < n_cols; j++) {
                if ((k < elr_next) && (A->col_idx[k] == j))  {
                    nnz_value = A->values[k];
                    res->values[i * n_cols + j] = logit * my_exp(nnz_value);
                    res->col_idx[i * n_cols + j] = j;
                    k++;
                } else {
                    res->values[i * n_cols + j] = logit;
                    res->col_idx[i * n_cols + j] = j;
                }
            }
        }

    // For other axes
    } else {

        for (j = 0; j < n_cols; j++) {
            // Compute the sum
            sum = n_cols;
            for (k = 0; k < n_nnz; k++) {
                if (A->col_idx[k] == j) {
                    nnz_value = A->values[k];
                    sum -= 1;
                    sum += my_exp(nnz_value);
                }
            }
            // Compute the logits
            logit = (double) (1.0 / sum);
            k = A->row_ptr[0];
            for (i = 0; i < n_rows; i++) {
                elr_next = A->row_ptr[i + 1];
                res->values[i * n_cols + j] = logit;
                res->col_idx[i * n_cols + j] = j;
                res->row_ptr[i + 1] += 1;
                while(k < elr_next) {
                    if (A->col_idx[k] == j) {
                        nnz_value = A->values[k];
                        res->values[i * n_cols + j] = logit * my_exp(nnz_value);
                    }
                    k++;
                }
            }
        }

    }
};

void softmax_csr_parallel(int axis, csr_matrix volatile *A, csr_matrix volatile *res, int core_id, int nPE) {

    int i, j, k;
    int n_rows, n_cols, n_nnz;
    double sum, logit, nnz_value;

    n_cols = A->cols;
    n_rows = A->rows;
    n_nnz = A->nnz;

    res->rows = n_rows;
    res->cols = n_cols;
    res->nnz = n_rows * n_cols;
    res->row_ptr[0] = 0;

    // For axis zero
    if (axis == 0) {
        k = 0;
        for (i = core_id; i < n_rows; i += nPE) {
            // Compute the sum
            sum = (A->cols) - ((A->row_ptr[i + 1]) - (A->row_ptr[i]));
            for (j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
                sum += my_exp(A->values[j]);
            }
            // Compute the Logits
            for (j = 0; j < A->cols; j++) {
                if ((k < A->row_ptr[i + 1]) && (A->col_idx[k] == j))  {
                    nnz_value = A->values[k];
                    res->values[i * n_cols + j] = my_exp(nnz_value) / sum;
                    res->col_idx[i * n_cols + j] = j;
                    k++;
                } else {
                    res->values[i * n_cols + j] = 1.0 / sum;
                    res->col_idx[i * n_cols + j] = j;
                }
            }
            res->row_ptr[i + 1] = res->row_ptr[i] + A->cols;
        }
        snrt_cluster_hw_barrier();
    // For other axes
    } else {
        for (j = core_id; j < A->cols; j += nPE) {
            // Compute the sum
            sum = (double) A->cols;
            for (k = 0; k < n_nnz; k++) {
                if (A->col_idx[k] == j) {
                    nnz_value = A->values[k];
                    sum += (my_exp(nnz_value) - 1);
                }
            }
            // Compute the logits
            logit = (double) (1.0 / sum);
            for (i = 0; i < n_rows; i++) {
                res->values[i * n_cols + j] = logit;
                res->col_idx[i * n_cols + j] = j;
                res->row_ptr[i + 1] += 1;
                k = A->row_ptr[i];
                while(k < A->row_ptr[i + 1]) {
                    if (A->col_idx[k] == j) {
                        nnz_value = A->values[k];
                        res->values[i * n_cols + j] = logit * my_exp(nnz_value);
                    }
                    k++;
                }
            }
        }
        snrt_cluster_hw_barrier();
    }
};
