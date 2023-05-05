
#include "host.c"
#include "data.h"

const axpy_args_t args = {L / 8, 2, (uint64_t)x, (uint64_t)y, (uint64_t)z};
const job_t axpy = {J_AXPY, .args.axpy = args};


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


    mcycle();
    comm_buffer.usr_data_ptr = (uint32_t)(uint64_t) & (axpy);
    // Start Snitches
    mcycle();
    wakeup_snitches_cl();
    // Wait for job done
    mcycle();
    
    wait_snitches_done();
    // Exit routine
    mcycle();
}