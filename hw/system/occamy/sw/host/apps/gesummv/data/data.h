#ifndef GESUMMV_DATA_H_
#define GESUMMV_DATA_H_

// Statically define the data which will be used for the computation
// (this will be loaded into DRAM together with the binary)

#define N 8
#define double double

const double alpha = 7.147349588510007;
const double beta = -9.218042442569022;

double A[N*N] = {
    -1.29927,  4.34593,  6.26424,  2.26448, -6.11240,  6.29630, -5.49063,  3.13199, 
    -4.90883, -9.40589, -0.85278,  0.90043, -9.53901,  1.39807, -8.79070, -5.37099, 
     9.15139, -7.44479, -7.17737,  5.23690,  9.48292,  6.16238,  9.82870, -3.11634, 
     7.81969, -1.35799, -7.74810,  3.83857, -2.37291, -5.04729, -6.09533, -4.84542, 
     7.15953, -1.57338, -6.45027, -8.34804,  3.11901, -2.07807,  9.91549, -9.66999, 
     7.68535, -2.68706, -4.15643,  7.15505,  9.32722, -3.97766,  9.64341, -1.65920, 
    -6.03529, -6.21325, -4.72512,  7.05652,  2.16334, -7.01376,  8.53003, -4.78659, 
     1.29422,  3.35788,  0.14088, -0.38144,  2.14839, -4.46367, -3.19579,  6.45258};


double B[N*N] = {
    -5.98467,  5.35813, -5.11748, -0.36180,  4.95764, -8.01046,  7.42145,  6.70859, 
    -1.63499, -3.93539, -6.25462,  2.83310,  1.24863,  2.91550, -1.84964, -4.68709, 
     6.81353,  9.87431,  6.35460, -5.43550, -2.43580,  9.50395,  2.91267, -3.92097, 
    -8.03443, -2.56187, -6.29995,  4.26570, -4.71871,  0.91486, -7.27267, -5.95645, 
    -7.19027, -0.71794,  3.78509, -1.33720,  8.60230, -5.44879,  4.14240, -9.88683, 
     4.11038, -1.96024,  2.36321, -0.46355,  3.87323,  2.16762, -2.18952, -0.87252, 
     7.36240, -9.94611, -7.87687, -7.06151,  4.33135, -3.10329,  9.50684, -8.05927, 
     4.41742,  5.82566,  3.88194, -6.21018,  7.83027,  9.87965, -8.76526, -0.70583};


double x[N] = {
    -7.02696, -2.14025,  4.98201,  3.26280,  2.87207, -9.91556,  6.76512,  5.58888};

double y[N];

uint32_t finished = 0;

#endif // GESUMMV_DATA_H_
