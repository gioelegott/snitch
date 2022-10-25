// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
\
<% def array_to_cstr(array):
    out = '{'
    for a in array:
        out += '{}, '.format(a)
    out = out[:-2] + '}'
    return out
%> \

typedef struct csr_matrix{
  double *data;
  double *indices;
  double *ptr;
  int len;
  int rows;
} csr_matrix;

csr_matrix A, B, C;

% for m, m_str in zip([A, B, C], ['A', 'B', 'C']):

// Data arrays for matrix ${m_str}
double ${m_str}_data[${m.nnz}] = ${array_to_cstr(m.data)};
double ${m_str}_indices[${m.nnz}] = ${array_to_cstr(m.indices)};
double ${m_str}_indptr[${m.shape[1]}] = ${array_to_cstr(m.indptr)};

// Array struct for matrix ${m_str}
${m_str}.data = ${m_str}_data;
${m_str}.indices = ${m_str}_indices;
${m_str}.indptr = ${m_str}_indptr;
${m_str}.len = ${m.nnz};
${m_str}.rows = ${m.shape[1]};

% endfor \
