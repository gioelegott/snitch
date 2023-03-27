// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#define N_JOBS 1

//////////
// AXPY //
//////////

#include "axpyTest.h"


/////////////
// Generic //
/////////////

typedef struct {
    uint64_t job_ptr;
} user_data_t;

typedef union {
    axpy_args_t axpy;
} job_args_t;

typedef struct {
    job_id_t id;
    job_args_t args;
} job_t;


