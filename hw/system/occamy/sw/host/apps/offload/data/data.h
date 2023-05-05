const uint32_t L = 16;

double a = 2;

double x_a[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

double y_a[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1};

double z_a[16];

#define N 8
#define N_g 8
#define N_l 8

#define double double

const double alpha = -3.506532936246021;
const double beta = -2.7470090466692643;

double A_g[N][N] = {
    {-4.94692, -9.09321,  6.89997, -4.32452, -9.69615, -2.06145,  3.11603,  8.39686},
    {-0.97547, -1.57941, -2.80009, -0.83144, -9.65350,  4.90643, -7.12321, -8.44164},
    { 5.68095, -7.54059, -3.41666,  1.59944, -6.05065,  8.70941, -0.80187,  0.32634},
    { 3.56032,  6.59875, -6.30726,  6.66589, -9.30983,  0.41064,  5.96930, -0.48311},
    { 9.76034,  0.37974,  4.89233, -6.13860, -6.42584,  9.44507, -9.03725,  5.19057},
    { 2.21838,  5.63060,  1.31690, -7.44186, -0.12377, -9.35931, -8.05504, -9.99695},
    {-2.16008, -5.38657, -7.02634,  1.33470,  8.77891, -5.55603, -0.93191, -9.89494},
    {-2.48189,  4.18862,  3.70264, -0.55100, -1.75845, -9.51739, -6.15763, -4.16327}};


double B_g[N][N] = {
    {-1.31217, -8.16445, -4.31336,  7.52830,  2.06925,  4.96430, -7.36154, -4.51837},
    { 1.82627,  5.57538,  2.06228,  7.12573,  2.63450, -0.81667, -2.12733, -6.24463},
    { 9.45021,  4.00540, -0.28231,  2.55947, -8.02963, -2.37853, -0.42630, -0.34068},
    { 2.09054,  7.51885,  6.27937,  7.11124,  7.87829, -9.53342,  8.49871,  3.00873},
    { 5.99391,  8.81911,  2.73631,  8.24802,  0.23744,  5.38681, -9.23194,  1.85040},
    { 6.49955,  3.63337, -5.28949,  2.21473, -7.78931, -2.57995,  4.07615,  8.18607},
    { 0.09292,  2.47301, -0.68858, -7.24117, -4.88580,  9.41558,  0.12073, -3.46386},
    { 4.62981,  7.78135, -2.52086, -4.53739,  8.75458,  9.43998, -7.73450,  0.03842}};


double x_g[N] = {-1.51062,  5.92457,  2.09063, -4.94660,  2.06227, -4.87423,  2.00946,  6.91471};

double y_g[N];

double A_l[N][N] = {
    {-4.94692, -9.09321,  6.89997, -4.32452, -9.69615, -2.06145,  3.11603,  8.39686},
    {-0.97547, -1.57941, -2.80009, -0.83144, -9.65350,  4.90643, -7.12321, -8.44164},
    { 5.68095, -7.54059, -3.41666,  1.59944, -6.05065,  8.70941, -0.80187,  0.32634},
    { 3.56032,  6.59875, -6.30726,  6.66589, -9.30983,  0.41064,  5.96930, -0.48311},
    { 9.76034,  0.37974,  4.89233, -6.13860, -6.42584,  9.44507, -9.03725,  5.19057},
    { 2.21838,  5.63060,  1.31690, -7.44186, -0.12377, -9.35931, -8.05504, -9.99695},
    {-2.16008, -5.38657, -7.02634,  1.33470,  8.77891, -5.55603, -0.93191, -9.89494},
    {-2.48189,  4.18862,  3.70264, -0.55100, -1.75845, -9.51739, -6.15763, -4.16327}};

