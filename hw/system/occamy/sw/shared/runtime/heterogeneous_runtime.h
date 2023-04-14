// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "occamy_memory_map.h"

// *Note*: to ensure that the usr_data field is at the same offset
// in the host and device (resp. 64b and 32b architectures)
// usr_data is an explicitly-sized integer field instead of a pointer
typedef struct {
    volatile uint32_t lock;
    volatile uint32_t usr_data_ptr;
} comm_buffer_t;


typedef enum { J_AXPY = 0, J_GESUMMV = 1, J_LU = 2 } job_id_t;

//////////
// AXPY //
//////////

typedef struct {
    uint32_t l;
    double a;
    uint64_t x_ptr;
    uint64_t y_ptr;
    uint64_t z_ptr;
} axpy_args_t;

typedef struct {
    uint32_t l;
    double a;
    uint64_t x_l3_ptr;
    uint64_t y_l3_ptr;
    uint64_t z_l3_ptr;
    double* x;
    double* y;
    double* z;
} axpy_local_args_t;

typedef struct {
    job_id_t id;
    axpy_args_t args;
} axpy_job_t;

typedef struct {
    job_id_t id;
    axpy_local_args_t args;
} axpy_local_job_t;


/////////////
// GESUMMV //
/////////////

typedef struct {
    uint32_t n;
    double alpha;
    double beta;
    uint64_t A_ptr;
    uint64_t B_ptr;
    uint64_t x_ptr;
    uint64_t y_ptr;
} gesummv_args_t;

typedef struct {
    uint32_t n;
    double alpha;
    double beta;
    uint64_t A_l3_ptr;
    uint64_t B_l3_ptr;
    uint64_t x_l3_ptr;
    uint64_t y_l3_ptr;
    double* A;
    double* B;
    double* x;
    double* y;
} gesummv_local_args_t;

typedef struct {
    job_id_t id;
    gesummv_args_t args;
} gesummv_job_t;

typedef struct {
    job_id_t id;
    gesummv_local_args_t args;
} gesummv_local_job_t;



////////
// LU //
////////

typedef struct {
    uint32_t n;
    uint64_t A_ptr;
} lu_args_t;

typedef struct {
    uint32_t n;
    uint64_t A_l3_ptr;
    double* A;
} lu_local_args_t;

typedef struct {
    job_id_t id;
    lu_args_t args;
} lu_job_t;

typedef struct {
    job_id_t id;
    lu_local_args_t args;
} lu_local_job_t;




/////////////
// Generic //
/////////////

typedef struct {
    uint64_t job_ptr;
} user_data_t;

typedef union {
    axpy_args_t axpy;
    gesummv_args_t gesummv;
    lu_args_t lu;
    
    //add args for new kernels here



} job_args_t;



typedef struct {
    job_id_t id;
    job_args_t args;
} job_t;






/**************************/
/* Quadrant configuration */
/**************************/

// Configure RO cache address range
inline void configure_read_only_cache_addr_rule(uint32_t quad_idx,
                                                uint32_t rule_idx,
                                                uint64_t start_addr,
                                                uint64_t end_addr) {
    volatile uint64_t* rule_ptr =
        quad_cfg_ro_cache_addr_rule_ptr(quad_idx, rule_idx);
    *(rule_ptr) = start_addr;
    *(rule_ptr + 1) = end_addr;
}

// Enable RO cache
inline void enable_read_only_cache(uint32_t quad_idx) {
    *(quad_cfg_ro_cache_enable_ptr(quad_idx)) = 1;
}
