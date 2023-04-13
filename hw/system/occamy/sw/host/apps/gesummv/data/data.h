#ifndef GESUMMV_DATA_H_
#define GESUMMV_DATA_H_

// Statically define the data which will be used for the computation
// (this will be loaded into DRAM together with the binary)

#define N 4
#define DATA_TYPE double

const DATA_TYPE alpha = -2.177098331274312;
const DATA_TYPE beta = 3.2435912601784764;

DATA_TYPE A[N*N] = {
     4.80024,  2.05905, -4.10045, -3.64988, 
     8.65375,  6.80253,  2.05059,  3.81319, 
     3.65861, -7.91251, -9.58467, -2.81182, 
    -6.44009, -9.41893, -0.96609, -7.70948};


DATA_TYPE B[N*N] = {
    -3.80397,  1.33258, -1.14248, -1.14050, 
     5.49336,  9.71487,  7.23122, -1.94488, 
     2.28821,  1.56035,  9.19088, -0.04801, 
    -7.24289,  9.09451, -1.89731,  4.17129};


DATA_TYPE x[N] = {-9.52416,  1.40156,  0.62336,  1.30063};

DATA_TYPE y[N];

uint32_t finished = 0;

#endif // GESUMMV_DATA_H_
