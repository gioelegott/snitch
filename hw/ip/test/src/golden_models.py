import numpy as np
import math
import re

# read input data for kernel computation from header file

def parse_arraysize(header_str, name):
	# find sizes of arrays
	arraysize_str = "#define " + name
	M_idx = header_str.find(arraysize_str) + 7 + len(name) + 2
	# find next #: length of number assigned to variable
	h_idx = header_str.find("#", M_idx)
	if h_idx == -1:
		h_idx = header_str.find("double", M_idx)
	M = int(header_str[M_idx:h_idx])
	return M

def parse_scalar(header_str, name):
	scalar_str = name + " ="
	alpha_idx = header_str.find(scalar_str)
	# find next semicolomn: length of number assigned to variable
	sc_idx = header_str.find(";", alpha_idx)
	alpha = float(header_str[alpha_idx+len(name)+3:sc_idx])
	return alpha

def parse_array(header_str, name, L):
	array_str = name + "["
	x_idx = header_str.find(array_str)
	c_idx = x_idx 
	entry_idx = x_idx+len(name)+7-3
	x = []
	for i in range(L-1):
		# find next comma: length of number
		entry_idx = entry_idx+3
		c_idx = header_str.find(",", c_idx + 1)
		x.append(int(header_str[entry_idx:c_idx]))
	# last element has no comma but }
	entry_idx = entry_idx+3
	c_idx = header_str.find("}", c_idx + 1)
	x.append(int(header_str[entry_idx:c_idx]))

	# convert to numpy array 
	x_np = np.array(x, dtype= np.double)
	return x_np

def parse_matrix(header_str, name, M, N):
	matrix_str = name + "["
	A_idx = header_str.find(matrix_str) 
	entry_idx = A_idx+len(name)+12-2
	c_idx = entry_idx
	A = [[] for i in range(M)]
	for j in range(M):
		for i in range(N-1):
			entry_idx = c_idx+2
			# find next comma: length of number
			c_idx = header_str.find(",", c_idx + 1)
			A[j].append(float(header_str[entry_idx:c_idx]))
		# last element has no comma but }
		entry_idx = c_idx+2
		c_idx = header_str.find("}", c_idx + 1)
		A[j].append(float(header_str[entry_idx:c_idx]))
		c_idx += 2 # there is a comma and additional bracket: },{
	# convert to numpy array 
	A_np = np.array(A, dtype= np.double)
	return A_np


def axpy(header_path):
	variable_name = "z"

	with open(header_path, "r") as header_file:
		header_str = header_file.read()

	# find array sizes
	L = parse_arraysize(header_str, "L")

	# find input variables
	a = parse_scalar(header_str, "a")
	x = parse_array(header_str, "x", L)
	y = parse_array(header_str, "y", L)

	# Calculate Kernel
	z = x*a + y
	
	return z, variable_name

def symm(header_path): 
	variable_name = "C" # same as named in the C code
	with open(header_path, "r") as header_file:
		header_str = header_file.read()

	# find array sizes
	M = parse_arraysize(header_str, "M")
	N = parse_arraysize(header_str, "N")

	# find input variables
	alpha = parse_scalar(header_str, "alpha")
	beta = parse_scalar(header_str, "beta")
	A = parse_matrix(header_str, "A", M, M)
	B = parse_matrix(header_str, "B", M, N)
	C = parse_matrix(header_str, "C", M, N)

	# Calculate Kernel
	for i in range(M):
		for j in range(N):
			temp2 = 0
			for k in range(i):
				C[k][j] += alpha*B[i][j]*A[i][k]
				temp2 += B[k][j]*A[i][k]
			C[i][j] = beta*C[i][j] + alpha*B[i][j]*A[i][i] + alpha*temp2
	
	return C, variable_name

def syrk(header_path): 
	variable_name = "C"
	with open(header_path, "r") as header_file:
		header_str = header_file.read()

	# find array sizes
	M = parse_arraysize(header_str, "M")
	N = parse_arraysize(header_str, "N")

	# find input variables
	alpha = parse_scalar(header_str, "alpha")
	beta = parse_scalar(header_str, "beta")
	A = parse_matrix(header_str, "A", N, M)
	C = parse_matrix(header_str, "C", N, N)

	# Calculate Kernel
	for i in range(N):
		for j in range(i+1):
			C[i][j] *= beta
		for k in range(M):
			for j in range(i+1):
				C[i][j] += alpha * A[i][k]*A[j][k]

	return C, variable_name

def cholesky(header_path): 
	variable_name = "A"
	with open(header_path, "r") as header_file:
		header_str = header_file.read()

	# find array sizes
	N = parse_arraysize(header_str, "N")

	# find input variables
	A = parse_matrix(header_str, "A", N, N)
	
	# Calculate Kernel
	for i in range(N):
		for j in range(i):
			for k in range(j):
				A[i][j] -= A[i][k]*A[j][k]
			A[i][j] /= A[j][j]
		for k in range(i):
			A[i][i] -= A[i][k]*A[i][k]
		A[i][i] = np.sqrt(A[i][i])

	return A, variable_name


def gramschmidt(header_path): 
	variable_name = ["Q", "R"]
	with open(header_path, "r") as header_file:
		header_str = header_file.read()

	# find array sizes
	M = parse_arraysize(header_str, "M")
	N = parse_arraysize(header_str, "N")

	# find input variables
	A = parse_matrix(header_str, "A", M, N)

	# Calculate Kernel: Output are matrices Q and R
	Q = np.zeros(shape=(M,N), dtype=np.double) 
	R = np.zeros(shape=(N,N), dtype=np.double) 

	# Modified Gram Schmidt by Walter Gander, 1980
	nrm = 0.0
	for k in range(N):
		nrm = 0.0
		for i in range(M):
			nrm += A[i][k] * A[i][k]
		R[k][k] = math.sqrt(nrm)
		for i in range(M):
			Q[i][k] = A[i][k] / R[k][k]
		for j in range(k+1, N):
		  R[k][j] = 0.0
		  for i in range(M):
		    R[k][j] += Q[i][k] * A[i][j]
		  for i in range(M):
		  	A[i][j] = A[i][j] - Q[i][k] * R[k][j]

	return [Q, R], variable_name
def gramschmidt(header_path): 
	variable_name = ["Q", "R"]
	with open(header_path, "r") as header_file:
		header_str = header_file.read()

	# find array sizes
	M = parse_arraysize(header_str, "M")
	N = parse_arraysize(header_str, "N")

	# find input variables
	A = parse_matrix(header_str, "A", M, N)

	# Calculate Kernel: Output are matrices Q and R
	Q = np.zeros(shape=(M,N), dtype=np.double) 
	R = np.zeros(shape=(N,N), dtype=np.double) 

	# Modified Gram Schmidt by Walter Gander, 1980
	nrm = 0.0
	for k in range(N):
		nrm = 0.0
		for i in range(M):
			nrm += A[i][k] * A[i][k]
		R[k][k] = math.sqrt(nrm)
		for i in range(M):
			Q[i][k] = A[i][k] / R[k][k]
		for j in range(k+1, N):
		  R[k][j] = 0.0
		  for i in range(M):
		    R[k][j] += Q[i][k] * A[i][j]
		  for i in range(M):
		  	A[i][j] = A[i][j] - Q[i][k] * R[k][j]

	return [Q, R], variable_name


def gesummv(header_path): 
	variable_name = ["y"]
	with open(header_path, "r") as header_file:
		header_str = header_file.read()

	# find array sizes
	N = parse_arraysize(header_str, "N")

	# find input variables
	alpha = parse_scalar(header_str, "alpha")
	beta = parse_scalar(header_str, "beta")
	
	A = parse_array(header_str, "A", N * N)
	B = parse_array(header_str, "B", N * N)
	x = parse_array(header_str, "x", N)
	y = parse_array(header_str, "y", N)


	for i in range(N):
		for j in range(M):
			y[i] += alpha * A[i*N + j] + beta * B[i*N + j]

	return y, variable_name
