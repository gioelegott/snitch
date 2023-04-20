#ifndef LU_DATA_H_
#define LU_DATA_H_

// Statically define the data which will be used for the computation
// (this will be loaded into DRAM together with the binary)

#define N 32
#define double double

double A[N*N] = {
     8.85255, -9.67169, -3.03039,  9.83512,  1.43876, -0.24693, -7.46935, -1.50681, -5.86053, -5.29490,  5.14179, -8.35721,  9.34153, -5.20703,  0.52587, -4.49297, -6.33698,  4.39633, -9.83685,  2.87349,  6.73951,  1.45401, -6.86666,  0.45155, -0.79943,  6.06588, -9.03235,  9.31718,  6.93689,  1.94799, -4.32893, -1.64241, 
     7.52163,  7.31423,  6.17373,  8.04729,  3.33815,  3.34387, -0.63202,  4.19911,  7.47196,  4.25727,  6.30120,  0.52869,  7.31605, -2.91318,  4.98886, -2.72023, -6.03381, -5.23912, -4.19593,  7.58999, -5.85146,  1.83080, -3.97919, -3.43800,  3.78470,  6.94937,  8.58122, -2.73183, -2.15219,  4.92221, -6.12868, -6.63084, 
    -3.47121, -2.66834, -6.94145, -0.21188,  9.73705, -6.07652, -5.25373, -2.75392,  0.53610, -3.26252,  0.91867,  1.96791, -5.03196, -3.16836, -4.74692,  8.08844,  4.60825,  6.12921,  8.29070, -3.47766,  1.53292, -1.91939, -1.24740, -8.17111,  9.51194,  1.55178, -2.35687,  5.55830,  1.86380, -0.70437, -3.94667, -2.48366, 
    -7.56264, -6.65848, -7.29432,  1.52535, -3.61947,  3.31437,  0.55104, -0.24455, -0.48100,  9.26688, -6.98519, -4.03490, -9.64017,  3.74674,  0.41487,  6.90823,  8.69078,  6.19038, -2.32929, -8.92307,  4.80495,  3.06229,  6.58143, -2.99138,  2.05199,  0.16408,  9.30255, -1.90296, -8.29887, -0.83987, -5.42535,  2.50744, 
     4.21750,  4.10528, -5.40337, -0.71816, -2.19720, -8.84703, -4.41661,  6.45913, -8.51105,  4.17074,  7.38915,  9.69091, -0.99761, -0.94732, -0.62244,  5.32429, -0.28931, -6.70971,  7.58565,  4.44398,  9.84044, -7.36644, -6.89788, -9.15290, -8.24575,  5.67924, -4.17725,  5.58706, -9.88108, -2.51040,  6.59662,  5.68383, 
     8.83736,  5.38311,  3.48802, -9.87758,  7.78343,  4.94419, -1.56332, -4.16834, -2.20462,  0.93471,  5.10500, -5.46917, -3.41477, -9.14148, -7.15093, -2.67750, -6.19891, -4.18702,  9.85398,  9.72704,  2.66218, -0.43830,  2.88005, -4.27748, -5.88892, -7.14991,  4.74352, -0.02317,  8.91554, -3.78287, -0.87703,  0.04643, 
    -2.00703, -4.02103, -9.63153,  3.48332,  9.92846, -0.40008, -0.64921,  1.51233, -7.10490, -1.34868, -4.58537,  6.68862,  9.31165,  1.12510, -8.77019,  7.31320,  9.66266, -8.27548, -8.66211, -1.94787,  9.03338, -4.46609,  5.50597, -0.39487, -1.66107,  0.52203,  3.84629,  8.26559, -4.34559, -5.05160, -6.04932,  2.48161, 
     8.02415,  7.35970,  9.68424, -6.44358,  3.20491,  1.45892,  3.65903,  0.20161, -9.53115,  4.90490,  2.03075, -8.48593,  5.40757, -7.87054,  7.92860, -2.23614, -8.61789,  1.59043, -4.78682,  6.19460,  9.31562,  8.17779,  0.82541,  1.39238,  5.51643, -1.72295,  7.42498, -6.07660,  2.21082,  0.84993, -2.02943,  1.59311, 
    -6.43903,  8.88062,  1.58545,  5.26693,  1.94172,  3.46983, -2.21341, -2.58227,  2.84111,  7.06934,  2.51719,  1.20661,  9.75876,  6.26498,  6.73204, -8.67863,  6.63014, -4.23423, -7.63860, -6.04081,  3.22582,  8.54826, -4.87771, -0.37499, -4.31458, -1.85190,  7.89547,  9.77916, -0.75040,  3.74868,  1.28122,  6.06315, 
    -8.21634,  5.25183, -5.93427, -8.69317, -8.61016, -5.40926, -0.01308, -6.71942,  4.68500,  7.34423,  3.73255, -9.42135, -9.40682,  4.80910, -7.27633, -8.92239, -9.26666, -9.17936,  1.79213,  7.28562, -5.62329,  1.81347, -8.87790,  8.62458, -2.83991, -1.16687, -6.30031,  9.35980,  8.72476, -8.09834,  9.19055,  3.88781, 
    -5.03988, -6.08377,  7.65266,  7.95126, -6.82813,  4.77478,  7.65940, -9.34738, -9.55927,  8.08130, -3.00671,  7.71271, -2.58083,  5.60025,  1.71416, -9.34209,  3.57478,  3.09841,  2.40908,  2.40380,  9.03480,  3.98060,  2.23023,  3.92177, -9.39654, -7.38200,  2.36017,  7.24786, -1.95025,  9.19692,  9.64249,  8.44146, 
     5.41851,  2.28507, -4.90715,  5.40247, -4.15583,  0.69404, -9.59577, -4.49908,  9.26660,  2.56098,  3.98936,  7.68049, -9.16494, -9.34992,  2.52979,  8.32655,  7.27074, -9.78288, -9.04182,  3.97724,  4.83255,  2.03812, -5.55302, -4.93605, -9.76383,  9.42415, -9.13709, -4.38324,  4.17777, -3.48348,  6.37717, -6.88360, 
    -4.17376, -6.50809, -9.22934, -5.09072, -4.20212, -9.27326, -1.80876, -7.09614,  2.38497,  0.30102, -8.77204,  7.58996,  6.61818, -3.16742,  9.99589,  3.27052, -0.53086, -0.39886, -9.45585,  6.24629, -2.96414,  4.79628, -6.99222,  3.45290,  3.24611, -4.04611, -4.19155, -9.95188,  9.02600,  1.05357,  9.65343, -5.29945, 
    -3.87297,  9.63072,  6.55269,  6.65835, -0.60450,  5.70758,  1.95981,  1.95160, -6.30114, -6.32338, -6.46418, -0.71425,  7.83665, -5.74340, -2.61257, -7.77028, -5.83171, -0.15163,  5.73452, -5.30125, -7.51341, -3.60014,  3.17792,  9.62809,  2.70891, -4.29699, -1.71075,  8.62127,  2.72271,  1.78347,  4.11519,  2.35086, 
     8.81551, -2.05145, -8.16972,  9.33709,  9.25677, -6.85824, -3.27978, -9.57708,  8.79323,  0.94664,  9.61311, -0.07531, -2.85063,  8.69675, -5.13946, -0.37303, -8.74318,  8.47245,  0.44805,  6.72060, -3.21241, -4.48486, -5.65697,  2.18664, -6.23976, -1.90084, -4.04103, -2.73539, -1.53966,  0.99170, -5.32327,  1.28726, 
     5.36925, -2.98823,  3.58262, -8.25894,  4.68619, -6.95346, -0.24597,  3.68997, -4.73377,  7.87401, -6.82915, -9.59801,  3.92502,  7.66308,  0.84115, -3.17881, -2.25052,  8.92776, -2.21098, -6.75424,  0.71783, -9.56548,  8.23704,  7.25304, -4.71711, -1.91900, -1.10354,  7.31921, -3.95742, -4.60195,  1.35823, -8.03255, 
    -1.09265, -8.14286, -2.50920,  8.05080,  2.73474,  8.94982,  4.83476,  2.56087,  0.88314,  3.41786, -4.07922,  2.11704,  5.68816, -7.87298,  0.18736,  3.12634, -5.93142, -2.43059, -4.98374,  3.19197,  7.15112,  2.58593,  4.50155, -8.95757, -3.72315, -7.79934, -2.05953,  3.29286,  3.16576,  6.46063, -2.71189, -4.23115, 
     7.51950, -1.81541,  0.07524,  2.39246, -7.38186,  2.14982,  3.04126,  4.07236, -2.89393, -7.68128,  9.70396,  7.83852, -4.58548, -8.26864,  2.79959,  9.13984, -5.44829, -7.70735, -5.74693, -9.30814,  9.25822, -4.92841, -9.79080,  5.13404, -4.61523,  3.11876, -0.48104,  9.97030, -7.65718, -2.72821,  8.60660,  1.35776, 
    -0.82247,  1.22574, -5.80027,  9.06929, -2.51096,  6.18668, -4.46002,  9.05147, -9.11452,  7.35408, -8.28768, -2.51626, -4.93979,  6.78586, -1.96770, -3.58094, -8.42619,  8.07860,  7.23217,  5.74511, -0.65507, -4.00955,  1.82164, -8.61206, -1.15270, -1.74936, -7.04553, -9.71776, -1.32446,  3.88177, -9.37778,  2.64731, 
    -8.26451,  9.70614, -7.71260, -0.43881,  8.19383, -0.28071, -6.61918,  5.63554,  5.61279,  2.19995,  3.30911, -6.41671, -9.66669, -7.19195,  9.58994,  8.32997,  4.77889,  8.28884,  0.23270, -3.86738, -9.49138,  5.03920,  1.75342, -8.93757,  3.86826,  9.41618, -6.18008,  1.09965,  6.32352, -0.93275, -6.55963, -1.68749, 
     0.85309,  4.64612, -8.86722, -9.68045, -6.57989,  4.57772, -2.03369, -9.92921, -0.23422,  4.16345,  8.01597,  4.49766, -5.57593,  4.48470,  2.92035,  9.81494,  6.03559, -8.33301, -6.72994,  7.23367,  2.70897, -4.84159, -6.17944, -2.64784,  8.96355,  3.53961,  6.95888,  5.11226, -2.50056, -7.32641,  3.08708, -0.64392, 
     0.64210,  7.99074,  1.87848, -8.49444, -9.37797,  9.51013, -3.67829, -2.47815, -2.42009,  9.33221,  9.07698,  0.86359,  7.90616,  7.88233,  8.51207,  4.73658, -1.91387,  5.38967, -2.05345, -2.79400,  6.63704, -3.71935, -8.94926, -5.09889, -6.43217,  2.80199,  0.51009,  7.31674, -0.80034,  0.86979, -6.70555,  1.50549, 
     8.74568,  7.66546, -0.68415, -8.82377,  9.24086,  8.32123,  2.63771, -4.46719,  6.83186,  8.67530,  2.03906,  4.41239, -9.23703,  0.07246, -8.90369,  7.78041, -0.59788,  3.83099,  6.65101, -1.64652, -8.14754, -1.44494, -5.71852,  1.88777, -1.31656,  3.65229,  0.24684,  6.20411, -2.96856,  3.64696,  4.22780, -7.28792, 
     3.85561,  3.88199,  0.99835, -9.09761,  1.49925, -5.18094,  1.88476, -8.48567,  6.52348, -0.83534, -7.18993,  6.33669, -0.74573,  4.13094, -5.79532, -3.20131,  0.31875,  3.90579,  7.31355,  2.94140, -5.87656,  7.21726, -5.85676,  4.19514,  4.35836,  6.40085,  4.62941, -7.03898,  6.96908,  8.52958, -4.14054,  9.67514, 
     0.11026,  2.42161, -1.48130,  0.48209, -4.10592, -3.42484, -4.42013,  4.23220, -3.23715, -9.47851,  3.04188, -7.29027,  6.31377, -1.99207, -0.41015, -7.51591, -0.61018, -0.42992, -0.45643,  9.03514, -5.26235,  9.28117, -6.05017, -8.58718, -2.76773, -2.53371, -4.17160, -7.35235,  0.39292,  2.54598, -4.63813, -1.72366, 
    -8.61319,  1.37121,  9.95368, -7.97869,  8.77882, -9.02223, -9.52512, -9.60036,  2.05224, -5.69714,  4.35740,  9.93468, -8.90436,  7.43393,  7.45278, -2.72069, -3.60575, -3.63336,  2.13236,  1.88919,  8.55535,  0.93881,  8.38156,  9.04546, -0.46099,  3.87659,  9.68444,  5.90582, -3.69689, -4.78743,  0.70674,  1.02137, 
     7.76830,  1.63966, -3.65629, -1.34782, -3.15976, -1.69694,  3.92644, -1.28091, -7.59747,  7.56245,  9.27200, -7.71995,  9.04759, -4.68970, -1.12914, -1.88931,  9.87488,  2.47783, -8.72508, -3.63185, -7.97928,  3.48429,  0.14773,  0.72893, -2.25949, -6.85191, -0.13157,  4.58601,  2.43324, -2.72071,  8.23631, -8.48690, 
     0.67148,  7.71435, -2.48378,  2.20343, -7.43524,  8.06042, -4.93409, -9.97342,  4.12921, -5.48798,  9.32236, -8.63355, -4.49809, -2.51360, -8.02772, -9.89017,  8.69130, -1.58264,  0.26190,  5.65530,  2.62953, -6.04682, -7.11435,  7.69126, -6.68009, -6.66910, -0.88839,  0.60913,  6.93400,  0.10045, -7.88213,  6.51572, 
     4.05859,  3.65907,  0.31756, -1.05881,  2.40523,  9.01855, -2.61499, -3.24596,  5.81251,  4.41643, -2.07099,  9.15036,  6.73176, -3.60023, -6.18612, -3.82792, -9.17421, -0.94863, -9.84386,  9.21143,  3.20070, -5.43280, -5.94109, -3.77325, -0.98502, -8.65459, -8.29792,  7.44354, -4.42233,  8.82562,  2.48387,  7.44022, 
    -5.04844, -9.14873, -0.94285,  3.79437, -1.64771,  5.79536, -2.40118,  5.92511,  0.63980,  8.03629, -1.95887,  6.79137, -9.51627, -3.54818, -4.47509,  4.19288,  0.11414, -7.87492, -4.00649,  5.60329,  0.96935,  7.21149, -3.77746,  3.89300, -5.34217,  3.99567,  5.53836, -6.98557, -3.10049,  2.58069,  2.34801, -7.72729, 
     0.60050, -6.90466, -1.43295,  7.63100, -6.61500, -3.05517,  3.82901, -0.85164, -1.74928,  9.07548,  7.40872,  4.77858,  1.65593, -5.20670, -6.57384, -5.92973,  6.79960,  1.70794, -2.44139, -7.59382,  4.59020, -4.24910,  9.80998, -8.37117, -9.83044,  8.30289,  3.05980,  1.33031,  9.50922, -5.65262, -5.62738, -1.72325, 
    -3.99229,  3.72302,  0.11500, -4.48336,  1.85952, -6.10001,  5.74936, -6.21030, -6.62735,  5.81949, -7.33919,  1.82892, -0.56716,  6.51559, -9.43573,  7.62466, -4.60214,  9.87662, -8.82884,  4.53103, -0.13948, -6.81121, -0.73519,  5.09450,  3.35595,  0.54940, -1.39788, -6.64265,  0.84909,  5.81090,  0.10506, -5.71770};


uint32_t finished = 0;

#endif // LU_DATA_H_
