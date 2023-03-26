#include "host.c"
#include "data.h"

void gesummv(uint32_t n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE *x, DATA_TYPE *y)
{
    uint32_t i, j;
    DATA_TYPE tmp1, tmp2;
  
    for (i = 0; i < n; i++)
    {
        tmp1 = tmp2 = 0;
        for (j = 0; j < n; j++)
        {
            tmp1 += alpha * A[i][j] * x[j];
            tmp2 += beta * B[i][j] * x[j];
        }
        y[i] = tmp1 + tmp2;
    }
}


int main() {
    // Wake up the Snitch cores even if we don't use them
    reset_and_ungate_quad(0);
    deisolate_quad(0, ISO_MASK_ALL);

    // Read the mcycle CSR (this is our way to mark/delimit a specific code region for benchmarking)
    uint64_t start_cycle = mcycle();
    
    // Call your kernel
    gesummv(N, alpha, beta, A, B, x, y);
    
    // Read the mcycle CSR
    uint64_t end_cycle = mcycle();
}
