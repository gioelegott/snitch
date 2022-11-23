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

#pragma once

#include "matrix_types.h"

///////////////////////////////////////////////////////////////////////
///////////////////////////     INPUT      ////////////////////////////
///////////////////////////////////////////////////////////////////////

csr_matrix A[${channel_size}];

% for i, m in enumerate(A):
// Data arrays for input matrix A[${i}]
/*
${m.todense()}
*/
double A${i}_data[${m.nnz}] = ${array_to_cstr(m.data)};
int A${i}_indices[${m.nnz}] = ${array_to_cstr(m.indices)};
int A${i}_indptr[${m.shape[1]+1}] = ${array_to_cstr(m.indptr)};
% endfor \

// Array struct for matrix A[${i}]
void assign_A(){
  if (snrt_cluster_core_idx() == 0){
% for i, m in enumerate(A):
    A[${i}] = (csr_matrix){A${i}_data, A${i}_indices, A${i}_indptr, ${m.nnz}, ${m.shape[0]}, ${m.shape[1]}};
% endfor \

  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////     FILTER      ///////////////////////////
///////////////////////////////////////////////////////////////////////

csr_matrix FILTER[${channel_size}];

% for i, m in enumerate(FILTER):
// Data arrays for input matrix FILTER[${i}]
/*
${m.todense()}
*/
double FILTER${i}_data[${m.nnz}] = ${array_to_cstr(m.data)};
int FILTER${i}_indices[${m.nnz}] = ${array_to_cstr(m.indices)};
int FILTER${i}_indptr[${m.shape[1]+1}] = ${array_to_cstr(m.indptr)};
% endfor \

// Array struct for matrix FILTER[${i}]
void assign_FILTER(){
  if (snrt_cluster_core_idx() == 0){
% for i, m in enumerate(FILTER):
    FILTER[${i}] = (csr_matrix){FILTER${i}_data, FILTER${i}_indices, FILTER${i}_indptr, ${m.nnz}, ${m.shape[0]}, ${m.shape[1]}};
% endfor \

  }
}

///////////////////////////////////////////////////////////////////////
//////////////////////////     RESULTS      ///////////////////////////
///////////////////////////////////////////////////////////////////////

csr_matrix RES;
% for m, m_str, m_name in zip([RES0], ['RES0'], ['RES[0]']):
// Data arrays for results ${m_str}
/*
${m.todense()}
*/
double ${m_str}_data[${m.nnz}] = ${array_to_cstr(m.data)};
int ${m_str}_indices[${m.nnz}] = ${array_to_cstr(m.indices)};
int ${m_str}_indptr[${m.shape[1]+1}] = ${array_to_cstr(m.indptr)};

% endfor \

// Array struct for matrix ${m_str}
void assign_RES(){
  if (snrt_cluster_core_idx() == 0){
% for m, m_str, m_name in zip([RES0], ['RES0'], ['RES[0]']):
    RES = (csr_matrix){${m_str}_data, ${m_str}_indices, ${m_str}_indptr, ${m.nnz}, ${m.shape[0]}, ${m.shape[1]}};
% endfor \

  }
}
