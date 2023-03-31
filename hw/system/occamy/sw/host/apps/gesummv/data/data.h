#ifndef GESUMMV_DATA_H_
#define GESUMMV_DATA_H_

// Statically define the data which will be used for the computation
// (this will be loaded into DRAM together with the binary)

#define N 16
#define DATA_TYPE double

const DATA_TYPE alpha = 0.4207033624110661;
const DATA_TYPE beta = -7.9398638740057415;

DATA_TYPE A[N][N] = {
    { 7.65503, -3.37973, -3.71904,  2.93730,  3.80362, -0.15095, -6.25783, -8.00074,  3.67300,  7.76201, -9.89089,  7.73278, -4.49435,  0.27337, -8.20111,  9.20503},
    {-7.07038, -0.88579,  1.82560, -8.31108,  6.99610,  5.33697,  8.64712, -5.49604,  1.05014,  0.66238,  3.78078,  4.90144, -6.94830,  6.61786, -7.32314, -5.58762},
    { 1.01225,  3.73704,  9.70227,  3.86367, -4.60074, -7.83418,  8.92823,  0.06660, -1.18545, -8.29778, -3.70047,  5.07519, -7.55240,  4.73863, -9.48798, -3.24823},
    { 2.98132,  1.17303,  2.00738,  7.48597, -9.75833, -0.71091, -6.71355, -7.96629, -8.62782, -4.74259, -3.18078,  5.06406, -0.55705, -9.16060,  0.57792,  3.46158},
    { 8.12150,  9.07740, -8.05122, -0.32458, -9.55650, -4.23740, -3.62185,  8.92165,  6.73217,  3.17783, -8.88831,  6.01790,  0.58483, -8.03560, -2.74020,  6.41271},
    {-1.43762,  0.89440,  2.39997,  9.43217,  5.87509, -1.51752, -5.56877,  5.45560, -7.37322, -8.29693,  6.97125, -3.00995, -9.64253,  3.88819,  3.72514, -4.90825},
    { 9.66341, -0.08684,  4.18995,  2.62750,  8.31820, -3.05600,  6.53202,  0.26643, -3.67658, -4.27126, -0.35395, -7.92957, -8.41603,  7.87151, -2.13001, -8.33696},
    { 3.41509, -6.98368, -1.53314, -1.31584,  1.80334, -0.81198,  3.69158,  5.86159, -0.89507,  4.74636,  4.20268, -2.30900,  5.21519,  3.89392, -6.78698, -4.40509},
    { 6.87660,  3.82773, -3.23776, -9.86621, -0.66364, -4.17143, -8.20520, -2.71982, -9.38999, -0.32990,  5.14147,  8.14202, -7.56593, -4.55553,  5.39044, -0.00057},
    { 7.16243, -0.93944,  6.48179,  5.93162,  3.25969,  2.91276,  9.71997,  0.58426, -9.99691,  3.11315,  7.14160, -8.90268, -2.73003, -1.09460, -6.82626,  7.98378},
    { 9.51947, -2.20379, -1.79271, -5.74843,  0.53170, -9.80732, -9.34383, -2.72388,  2.04389,  7.68353, -0.44171, -6.33760, -4.90083,  7.52509, -7.72036, -6.62849},
    {-4.13614,  6.27641, -2.20157, -0.11516,  3.13647, -3.39674, -2.10075,  9.38439, -7.65058, -6.33946,  5.17753, -7.31512, -5.11365, -6.81663, -5.69401, -1.60936},
    {-1.84987,  3.00257, -0.54473, -2.04128,  5.41798,  7.28042,  5.98107, -8.24726,  9.97374, -7.22841,  5.46791,  6.69474, -1.58165,  8.32027,  8.73634, -6.30276},
    {-4.53078, -7.95672,  7.35396, -5.54494,  5.65522, -1.11237, -1.94390,  6.43059, -6.22580,  9.70769, -8.38891, -7.03785,  9.04668, -5.01858, -1.97125,  1.77605},
    { 8.21756, -9.50744, -8.93583,  6.47772, -8.68861, -4.44385,  3.19008,  6.66620, -4.17497,  5.91078,  9.84992,  5.07186, -2.34098,  4.28891, -8.92997,  4.52598},
    {-7.50939,  4.72855,  2.26844,  1.01185,  8.68721,  0.47953, -5.63452,  3.91511, -6.95779, -5.63625, -5.70835, -0.13378,  7.22040, -1.87750, -4.92108, -1.81966}};


DATA_TYPE B[N][N] = {
    {-1.37673,  1.14869,  4.74578, -1.83992,  7.33247,  2.26631,  7.33074,  4.03433,  6.00923, -2.93334, -6.04332, -9.33234,  5.84420,  9.29757, -5.02230, -3.84791},
    { 0.66464,  4.90437,  9.31123, -7.13623, -0.52561,  5.98772,  7.86760,  7.25217,  6.49671,  2.24228, -5.18322,  6.14446,  0.15702,  6.93110,  6.67364, -7.79681},
    { 2.39192,  1.59235, -7.61949,  7.19897, -6.71649, -0.74091,  0.72391,  4.91485, -8.83501, -9.06801, -0.22431, -0.43789, -8.71962,  5.44658,  8.84049, -7.00425},
    {-1.56868, -8.88636,  1.75767, -4.23360,  2.92264, -0.91482,  3.63727,  2.37744, -8.49462, -0.45309,  3.85911,  9.41924,  8.16950, -9.20324,  6.29271, -9.41794},
    {-9.55918,  6.91760,  1.76707,  2.61920, -3.19048, -2.58481,  8.77087,  9.82206,  7.12098, -2.93519,  9.83582,  7.85798, -6.34857, -1.41110,  1.27407,  5.58570},
    {-7.46342,  9.06968, -6.26295,  6.21991,  4.11881,  6.52870, -0.38404,  8.73953, -6.91225, -2.06811, -8.55351, -9.45558,  8.26223, -9.70945, -2.65179, -5.15906},
    { 0.09666, -5.36803, -7.94300,  5.09363, -4.14958, -8.09798, -8.93885, -4.78004, -3.25239, -3.32734, -8.14214, -8.30580,  4.31111,  3.98932,  3.33583,  9.95305},
    { 3.98055,  2.44512, -2.42130, -0.67392,  1.06598,  7.39988,  4.25666,  1.41423,  1.85165,  9.61137, -4.71502, -4.35139, -1.62137,  3.56041,  3.68083, -2.74064},
    {-6.62962, -5.83440, -4.12214, -6.68211,  3.80570, -9.16946,  4.59622,  7.15054,  1.77113, -4.66071,  1.89629,  3.72570,  7.92767,  6.02372, -1.80165, -7.14872},
    { 8.17252, -2.33590, -2.97738, -2.67860, -0.46702,  5.07887,  4.96362, -4.31008,  4.43990,  0.48433,  3.16065,  4.09545, -5.97385, -9.09585, -4.87461, -2.03269},
    {-5.37838, -0.65857, -4.34904,  8.62857,  6.95962, -9.51829, -5.77529,  3.89946,  1.85850, -4.40894, -3.13671,  4.52406,  3.76533, -5.54689,  4.45268,  2.30299},
    { 4.04536, -3.16814,  8.73183,  3.85243,  6.95407,  7.65393, -2.17523,  8.74961, -7.01667,  1.84108,  4.63339,  4.38297, -8.26794,  7.37729, -0.21437,  7.55336},
    { 4.31402,  8.72230,  4.49807,  7.08197, -3.67977, -3.71546,  2.49588,  1.32507, -3.12024,  8.07208,  6.81046, -4.13107,  4.31879, -9.76615, -7.82381, -2.31909},
    {-9.00723,  9.77398,  2.07202, -9.38471, -2.40286,  8.65689, -4.25126, -7.37327,  7.68658,  1.13409, -7.98287,  7.16616,  4.70063,  9.07362,  2.29296,  6.57248},
    {-7.13839, -6.18194,  9.36684, -3.89373, -7.85882, -4.27511,  7.60848, -4.24677, -2.25802,  7.87939,  7.40460, -1.19250, -9.73852, -9.53278,  2.03243,  1.08791},
    { 4.80334, -4.22528,  2.73200, -8.69442,  6.61729, -4.45508, -0.28067,  1.72146,  1.39380,  6.88427,  5.70856,  0.07055, -5.77099,  5.60062,  8.52460,  9.46673}};


DATA_TYPE x[N] = { 0.86967,  0.01145, -2.81110, -7.10948,  4.21778,  9.85061, -6.74630,  0.63904,  1.03570, -9.48213,  6.21771,  8.48667,  7.68550, -4.49787,  2.91723, -5.67912};

DATA_TYPE y[N];

uint32_t finished = 0;

#endif // GESUMMV_DATA_H_
