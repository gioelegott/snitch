
#include "snrt.h"
#include "luMulticluster.h"

#define CEIL(x, y) ((((x) - 1) / (y)) + 1)
#define MIN(x, y) ((x) < (y)?(x):(y))
#define double double

__thread volatile comm_buffer_t* comm_buffer_luMulticluster;

void luMulticluster(uint32_t n, double **A)
{
    uint32_t i, j, k;
    double tmp;

    uint32_t local_core_idx = snrt_cluster_core_idx();
    uint32_t local_core_num = snrt_cluster_compute_core_num();
    uint32_t global_core_idx = snrt_global_core_idx();
    uint32_t global_core_num = snrt_global_core_num();
    uint32_t cluster_idx = snrt_cluster_idx();
    uint32_t cluster_num = snrt_cluster_num();



    for (k = 0; k < n; k++)
    {
        double akk = A[(k/2)%2][k*n + k];
        uint32_t kk = k + 1;

        uint32_t c = CEIL((n/2), global_core_num);
        int32_t lb = (c * local_core_idx * 2 + cluster_idx)*2;
        int32_t ub = MIN((c * 2 * (local_core_idx + 1) + cluster_idx)*2, n);

        for (i = lb; i < ub; i+=4)
        {
            A[cluster_idx][i * n + k] /= akk;
            A[cluster_idx][(i + 1) * n + k] /= akk;
        }

        sw_barrier();

        //fetching data from other clusters

        sw_barrier();

        //if this doesn't work we can write the updated value in both TCDMs -> directly compute the final value using frep and then copy to both locations
        for (i = kk + ((global_core_idx/4 - kk%2 +2)%2); i < n; i += 2)
	        for (j = kk + ((global_core_idx%4 - kk%4 +4)%4); j < n; j += 4)
	            A[cluster_idx][i*n + j] = A[cluster_idx][i*n + j] - A[cluster_idx][i*n + k] * A[cluster_idx][k*n + j];
        sw_barrier();

    }


    // for (k = 0; k < n; k++)
    // {
    //     mcycle();
    //     for (i = k + 1 + core_idx; i < n; i += core_num)
	//         A[i*n + k] = A[i*n + k] / A[k*n + k];

    //     mcycle();
    //     snrt_cluster_hw_barrier();
    //     mcycle();

    //     for (i = k + 1 + core_idx; i < n; i += core_num)
	//         for (j = k + 1; j < n; j++)
	//             A[i*n + j] = A[i*n + j] - A[i*n + k] * A[k*n + j];
    //     mcycle();
    //     snrt_cluster_hw_barrier();
    //     mcycle();

    // }


    snrt_fpu_fence();
}


void luMulticluster_job_dm_core(job_t* job) {

    // Get local job pointer as next free slot in l1 alloc
    luMulticluster_local_job_t* luMulticluster_job = (luMulticluster_local_job_t*)snrt_l1_next();

    mcycle(); //2|3

    // Copy job info
    snrt_dma_start_1d(luMulticluster_job, job, sizeof(luMulticluster_job_t));
    snrt_dma_wait_all();
    uint32_t n = luMulticluster_job->args.n;
    size_t matrix_size = n * n * 8;

    // Copy operand A
    double* A = (double*)((uint32_t)luMulticluster_job + sizeof(luMulticluster_local_job_t));
    snrt_dma_start_1d(A, (void*)(uint32_t)luMulticluster_job->args.A_l3_ptr, matrix_size);

    uint32_t cluster_idx = snrt_cluster_idx();
    uint32_t cluster_num = snrt_cluster_num();

    // Send local pointer to memory so that all cluster can access it
    ((double**)(job->args.luMulticluster.A_ptr_clusters))[cluster_idx] = A;

    sw_barrier();

    double* A_l1_ptr[cluster_num];
    for (int i = 0; i < cluster_num; i++)
    {
        if (i == cluster_idx)
            luMulticluster_job->args.A[i] = A;
        else
            luMulticluster_job->args.A[i] = ((double**)(job->args.luMulticluster.A_ptr_clusters))[cluster_idx];

        A_l1_ptr[i] = luMulticluster_job->args.A[i];
    }
    // Wait for DMA transfers to complete
    snrt_dma_wait_all();

    mcycle(); //3|4

    snrt_cluster_hw_barrier();

    // Now we can update the L1 alloc pointer
    void* next = (void*)((uint32_t)(luMulticluster_job->args.A) + matrix_size);
    snrt_l1_update_next(next);

    mcycle(); //4|5
    // Synchronize cores to make sure results are available before
    // DMA starts transfer to L3

    for (uint32_t k = 0; k < n; k++)
    {
        sw_barrier();
        //if data is not in this TCDM retreive it from the other cluster
        if ((k/2)%2 != cluster_idx)
        {
            snrt_dma_start_1d(A_l1_ptr[cluster_idx] + k * n + k, A_l1_ptr[(cluster_idx + 1)%cluster_num] + k * n + k, (n - k) * sizeof(double));
            snrt_dma_wait_all();
        }

        sw_barrier();
        sw_barrier();
    }

    mcycle(); //5|6

    // Transfer data out 2 rows at the time
    for (int i = cluster_idx * 2; i < n; i+=4)
        snrt_dma_start_1d((void*)(uint32_t)luMulticluster_job->args.A_l3_ptr, A_l1_ptr[cluster_idx] + i * n, n * 2 * sizeof(double));
    snrt_dma_wait_all();

    mcycle(); //6|7


}




void luMulticluster_job_compute_core(job_t* job) {

    // Synchronize with DM core to wait for operands
    // to be fully transferred in L1
    //snrt_cluster_hw_barrier();

    // Cast local job
    luMulticluster_local_job_t* luMulticluster_job = (luMulticluster_local_job_t*)job;

    // Get args
    luMulticluster_local_args_t args = luMulticluster_job->args;
    uint32_t n = args.n;
    double** A = args.A;

    // Run kernel
    mcycle();//4|5
    luMulticluster(n, A);
    mcycle();//5|6


    // Synchronize with DM core to make sure results are available
    // before DMA starts transfer to L3
    snrt_cluster_hw_barrier();

    mcycle();//6|7


}



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
    job_t* job_remote = ((job_t*)comm_buffer_luMulticluster->usr_data_ptr);

    // Invoke job
    luMulticluster_job_dm_core(job_remote);

    // Synchronize clusters
    uint32_t cnt = snrt_sw_barrier_arrival();
    if (cnt == snrt_cluster_num()) {
        snrt_reset_barrier();
        snrt_int_sw_set(0);
    }
    mcycle(); //7|8
    goto run_job_end;

run_job_compute_core:;

    // Get pointer to local copy of job
    job_t* job_local = (job_t*)(snrt_l1_next());

    mcycle();//2|3

    // Synchronize with DM core such that it knows
    // it can update the l1 alloc pointer, and we know
    // job data is locally available
    snrt_cluster_hw_barrier();

   mcycle();//3|4

    // Invoke job
    luMulticluster_job_compute_core(job_local);

run_job_end:;
}


__attribute__((weak)) int main() {

    comm_buffer_luMulticluster = (volatile comm_buffer_t*)get_communication_buffer();

    // Notify CVA6 when snRuntime initialization is done
    post_wakeup_cl();
    return_to_cva6(SYNC_ALL);
    snrt_wfi();

    // Reset state after wakeup
    mcycle(); //0|1
    post_wakeup_cl();

    // Execute job
    mcycle(); //1|2
    run_job();

    return_to_cva6(SYNC_ALL);
}

