#include "snrt.h"
#include "data.h"

// Define your kernel
void axpy(uint32_t l, double a, double *x, double *y, double *z) {
    for (uint32_t i = 0; i < l ; i++) {
        z[i] = a * x[i] + y[i];
    }
}

int main() {

    // Perform some operations (e.g. clear interrupt) after wakeup
    post_wakeup_cl();

    // DM core does not participate in the computation
    if(snrt_is_compute_core()) {
        uint32_t start_cycle = mcycle();
        axpy(L, a, x, y, z);
        uint32_t end_cycle = mcycle();
    }

    // Synchronize all cores and send an interrupt to CVA6
    return_to_cva6(SYNC_ALL);
}

// #include "snrt.h"
// #include "data.h"

// // Define your kernel
// void axpy(uint32_t l, double a, double *x, double *y, double *z) {
//     int core_idx = snrt_cluster_core_idx();
//     int offset = core_idx * l;

//     for (int i = 0; i < l; i++) {
//         z[offset] = a * x[offset] + y[offset];
//         offset++;
//     }
//     snrt_fpu_fence();
// }

// int main() {

//     if(snrt_is_dm_core())
//         return 0;

//     uint32_t start_cycle = mcycle();
//     axpy(L / snrt_cluster_compute_core_num(), a, x, y, z);
//     uint32_t end_cycle = mcycle();

//     return 0;
// }