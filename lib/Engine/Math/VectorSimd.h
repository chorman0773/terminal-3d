#ifndef VECTOR_SIMD_H_2023_09_11_22_36_29
#define VECTOR_SIMD_H_2023_09_11_22_36_29

#if defined(__x86_64__)||defined(__i386__)
#include "bits/VectorSimdSSE.h"
#else
#include "bits/VectorScalar.h"
#endif

#endif /* VECTOR_SIMD_H_2023_09_11_22_36_29 */
