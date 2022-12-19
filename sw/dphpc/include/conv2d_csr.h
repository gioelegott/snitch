// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yichao Zhang <yiczhang@iis.ee.ethz.ch>

#pragma once

#include "matrix_types.h"

void conv2d_csr(csr_matrix **A, csr_matrix **filter, csr_matrix *res, int channel_in, int filter_row, int res_row, int res_col);
void conv2d_dense(dense_matrix **A, dense_matrix **filter, csr_matrix *res, int channel_in, int A_col, int filter_row, int filter_col, int res_row, int res_col);
void conv2d_csr_dense(csr_matrix **A, dense_matrix **filter, csr_matrix *res, int channel_in, int filter_row, int filter_col, int res_row, int res_col);