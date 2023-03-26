
//https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1
//https://github.com/cavazos-lab/PolyBench-ACC


#include "host.c"
#include "data.h"


void lu(uint32_t n, DATA_TYPE A[N][N])
{
    uint32_t i, j, k;
    DATA_TYPE tmp;

    /*standard*/
    // for (i = 0; i < n; i++) {

    //     for (j = 0; j <i; j++) {
    //         for (k = 0; k < j; k++) {
    //             A[i][j] -= A[i][k] * A[k][j];
    //         }
    //         A[i][j] /= A[j][j];
    //     }

    //     for (j = i; j < n; j++) {
    //         for (k = 0; k < i; k++) {
    //             A[i][j] -= A[i][k] * A[k][j];
    //         }
    //     }
        
    // }


    /*registers*/
    for (i = 0; i < n; i++) {

        for (j = 0; j < i; j++) {
            tmp = A[i][j];
            for (k = 0; k < j; k++) {
                tmp -= A[i][k] * A[k][j];
            }
            A[i][j] = tmp / A[j][j];
        }
        //__rt_get_timer();

        for (j = i; j < n; j++) {
            tmp = A[i][j];
            for (k = 0; k < i; k++) {
                tmp -= A[i][k] * A[k][j];
            }
            A[i][j] = tmp;
        }
        
    }


    /*polybench acc*/
    // for (k = 0; k < n; k++)
    // {
    //     for (j = k + 1; j < n; j++)
    //         A[k][j] = A[k][j] / A[k][k];

    //     for(i = k + 1; i < n; i++)
    //         for (j = k + 1; j < n; j++)
    //             A[i][j] = A[i][j] - A[i][k] * A[k][j];
    // }


    
    // /*A[i*n+j]*/
    // for (i = 0; i < n; i++) {

    //     for (j = 0; j <i; j++) {
    //         tmp = A[i*n + j];
    //         for (k = 0; k < j; k++) {
    //             tmp -= A[i*n + k] * A[k*n + j];
    //         }
    //         A[i*n + j] = tmp / A[j*n + j];
    //     }
    //     //__rt_get_timer();

    //     for (j = i; j < n; j++) {
    //         tmp = A[i*n + j];
    //         for (k = 0; k < i; k++) {
    //             tmp -= A[i*n + k] * A[k*n + j];
    //         }
    //         A[i*n + j] = tmp;
    //     }
        
    // }

    /*gaussian reduction*/

    // for (i = 0; i < n-1; i++)
    // {
    //     for (j = i+1; j < n; j++)
    //     {
    //         tmp = A[j][i]/A[i][i];
    //         for (k = i; k < n; k++)
    //             A[j][k] -= A[i][k] * tmp;

    //         A[j][i] = tmp;
    //     }
    // }



}

int main() {
    // Wake up the Snitch cores even if we don't use them
    reset_and_ungate_quad(0);
    deisolate_quad(0, ISO_MASK_ALL);

    // Read the mcycle CSR (this is our way to mark/delimit a specific code region for benchmarking)
    uint64_t start_cycle = mcycle();
    
    // Call your kernel
    lu(N, A);
    
    // Read the mcycle CSR
    uint64_t end_cycle = mcycle();
}



