// #include "host.c"
// #include "data.h"

// // Define your kernel
// void axpy(uint32_t l, double a, double *x, double *y, double *z) {
//     for (uint32_t i = 0; i < l ; i++) {
//         z[i] = a * x[i] + y[i];
//     }
// }

// int main() {
//     // Wake up the Snitch cores even if we don't use them
//     reset_and_ungate_quad(0);
//     deisolate_quad(0, ISO_MASK_ALL);

//     // Read the mcycle CSR (this is our way to mark/delimit a specific code region for benchmarking)
//     uint64_t start_cycle = mcycle();
    
//     // Call your kernel
//     axpy(L, a, x, y, z);
    
//     // Read the mcycle CSR
//     uint64_t end_cycle = mcycle();
// }
//  

#include "host.c"

int main() {
    // Reset and ungate quadrant 0, deisolate
    reset_and_ungate_quad(0);
    deisolate_quad(0, ISO_MASK_ALL);

    // Enable interrupts to receive notice of job termination
    enable_sw_interrupts();

    // Program Snitch entry point and communication buffer
    program_snitches();

    // Wakeup Snitches with interrupts
    wakeup_snitches_cl();

    // Wait for an interrupt from the Snitches to communicate that they are done
    wait_snitches_done();
}