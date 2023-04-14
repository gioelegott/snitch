import sys
import numpy as np

lower = -10
upper = 10

width = 8

# Check that the correct number of arguments have been passed
if len(sys.argv) != 3:
    print("Usage: python generate_header.py [dimension] [data_type]")
    sys.exit(1)

# Get the integer and string arguments from the command line
try:
    N = int(sys.argv[1])
except ValueError:
    print("Error: First argument must be an integer")
    sys.exit(1)

DATA_TYPE_NAME = sys.argv[2]
DATA_TYPE = np.dtype(sys.argv[2])
# Generate the arrays based on the integer and string arguments
arr1 = np.random.uniform(lower, upper, size=(N * N)).astype(DATA_TYPE)

# Open the header file for writing
with open("data/data.h", "w") as f:
    # Write the header file preamble
    f.write("#ifndef LU_DATA_H_\n")
    f.write("#define LU_DATA_H_\n\n")
    f.write("// Statically define the data which will be used for the computation\n")
    f.write("// (this will be loaded into DRAM together with the binary)\n\n")
    
    # Write the N and DATA_TYPE constants
    f.write("#define N {}\n".format(N))
    f.write("#define DATA_TYPE {}\n\n".format(DATA_TYPE_NAME))
    
    # Write the arrays
    
    # f.write("DATA_TYPE A[N][N] = {\n    {")
    # for i in range(len(arr1)-1):
    #     if (i+1) % N == 0:
    #         f.write("{0: >{width}.5f}".format(arr1[i], width=width))
    #         f.write("},\n    {")
    #     else:
    #         f.write("{0: >{width}.5f}, ".format(arr1[i], width=width))
    # f.write("{0: >{width}.5f}}}}};\n\n\n".format(arr1[-1], width=width))

    
    f.write("DATA_TYPE A[N*N] = {\n    ")
    for i in range(len(arr1)-1):
        f.write("{0: >{width}.5f}, ".format(arr1[i], width=width))
        if (i+1) % N == 0:
            f.write("\n    ")
    f.write("{0: >{width}.5f}}};\n\n\n".format(arr1[-1], width=width))

    
        
    f.write("uint32_t finished = 0;\n\n")
    
    f.write("#endif // LU_DATA_H_\n")
    
# Print a message to indicate success
print("Header file lu.h generated successfully!")

