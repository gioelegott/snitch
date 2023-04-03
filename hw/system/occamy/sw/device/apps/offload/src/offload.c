// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"
#include "axpy.h"
#include "gesummv.h"

#define N_JOBS 2

// Job function type
typedef void (*job_func_t)(job_t* job);

// Job function arrays
__thread job_func_t jobs_dm_core[N_JOBS] = {axpy_job_dm_core, gesummv_job_dm_core};
__thread job_func_t jobs_compute_core[N_JOBS] = {axpy_job_compute_core, gesummv_job_compute_core};

// Other variables
__thread volatile comm_buffer_t* comm_buffer;


static inline void run_job() {
    // Force compiler to assign fallthrough path of the branch to
    // the DM core. This way the cache miss latency due to the branch
    // is incurred by the compute cores, and overlaps with the data
    // movement performed by the DM core.
    asm goto("bnez %0, %l[run_job_compute_core]"
             :
             : "r"(snrt_is_compute_core())
             :
             : run_job_compute_core);

    // Retrieve remote job data pointer
    job_t* job_remote = ((job_t*)comm_buffer->usr_data_ptr);

    // Invoke job
    jobs_dm_core[job_remote->id](job_remote);

    // Synchronize clusters
    uint32_t cnt = snrt_sw_barrier_arrival();
    if (cnt == snrt_cluster_num()) {
        snrt_reset_barrier();
        snrt_int_sw_set(0);
    }

    goto run_job_end;

run_job_compute_core:;

    // Get pointer to local copy of job
    job_t* job_local = (job_t*)(snrt_l1_next());

    mcycle();

    // Synchronize with DM core such that it knows
    // it can update the l1 alloc pointer, and we know
    // job information is locally available
    snrt_cluster_hw_barrier();

    mcycle();

    // Invoke job
    jobs_compute_core[job_local->id](job_local);

run_job_end:;
}

int main() {
    // Enable RO cache on whole HBM address space
    // if (snrt_is_dm_core()) {
    //     configure_read_only_cache_addr_rule(snrt_quadrant_idx(), 0,
    //                                         HBM_00_BASE_ADDR,
    //                                         HBM_10_BASE_ADDR);
    //     enable_read_only_cache(snrt_quadrant_idx());
    // }

    // Initialize pointers
    comm_buffer = (volatile comm_buffer_t*)get_communication_buffer();

    // Notify CVA6 when snRuntime initialization is done
    post_wakeup_cl();
    return_to_cva6(SYNC_ALL);
    snrt_wfi();

    // Job loop
    while (1) {
        // Reset state after wakeup
        mcycle();
        post_wakeup_cl();

        // Execute job
        mcycle();
        run_job();

        // Go to sleep until next job
        mcycle();
        snrt_wfi();
    }
}
