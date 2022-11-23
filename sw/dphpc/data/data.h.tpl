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

csr_matrix A[2];

% for m, m_str, m_name in zip([A0, A1], ['A0', 'A1'], ['A[0]', 'A[1]']):
// Data arrays for input matrix ${m_str}
/*
${m.todense()}
*/
double ${m_str}_data[${m.nnz}] = ${array_to_cstr(m.data)};
int ${m_str}_indices[${m.nnz}] = ${array_to_cstr(m.indices)};
int ${m_str}_indptr[${m.shape[1]+1}] = ${array_to_cstr(m.indptr)};
% endfor \

// Array struct for matrix ${m_str} 
void assign_A(){
  if (snrt_cluster_core_idx() == 0){
% for m, m_str, m_name in zip([A0, A1], ['A0', 'A1'], ['A[0]', 'A[1]']):
    ${m_name} = (csr_matrix){${m_str}_data, ${m_str}_indices, ${m_str}_indptr, ${m.nnz}, ${m.shape[0]}, ${m.shape[1]}};
% endfor \

  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////     FILTER      ///////////////////////////
///////////////////////////////////////////////////////////////////////

csr_matrix FILTER[2];

% for m, m_str, m_name in zip([FILTER0, FILTER1], ['FILTER0', 'FILTER1'], ['FILTER[0]', 'FILTER[1]']):
// Data arrays for filter ${m_str}
/*
${m.todense()}
*/
double ${m_str}_data[${m.nnz}] = ${array_to_cstr(m.data)};
int ${m_str}_indices[${m.nnz}] = ${array_to_cstr(m.indices)};
int ${m_str}_indptr[${m.shape[1]+1}] = ${array_to_cstr(m.indptr)};
% endfor \

// Array struct for matrix ${m_str}
void assign_B(){
  if (snrt_cluster_core_idx() == 0){
% for m, m_str, m_name in zip([FILTER0, FILTER1], ['FILTER0', 'FILTER1'], ['FILTER[0]', 'FILTER[1]']):
    ${m_name} = (csr_matrix){${m_str}_data, ${m_str}_indices, ${m_str}_indptr, ${m.nnz}, ${m.shape[0]}, ${m.shape[1]}};
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
