#pragma once

#include <iostream>
#include <cstdlib>

// May be user-parameterizable, default On on Debug and Off on Release
#ifndef CUDA_EXIT_ON_ERROR
#   ifdef NDEBUG
// Release
#       define CUDA_EXIT_ON_ERROR 0
#   else
// Debug
#       define CUDA_EXIT_ON_ERROR 1
#   endif
#endif

// May be user-parameterizable, default 256 threads as a (16, 16) block
#ifndef CUDA_NUM_THREAD
#   define CUDA_NUM_THREAD dim3(16, 16)
#endif

// Error checking
#define CUDA_CHECK(val) check_cuda_error((val), #val, __FILE__, __LINE__)
template <class T>
void check_cuda_error(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;

#       if CUDA_EXIT_ON_ERROR
            std::exit(EXIT_FAILURE);
#       endif
    }
}

// Check last CUDA call
#define CUDA_CHECK_LAST() check_last_cuda_error(__FILE__, __LINE__)
inline void check_last_cuda_error(const char* const file, const int line)
{
    const cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;

#       if CUDA_EXIT_ON_ERROR
            std::exit(EXIT_FAILURE);
#       endif
    }
}

template<class T>
inline T ceil_div(T num, T den)
{
    return (num + den - static_cast<T>(1)) / den;
}

inline dim3 ceil_div(dim3 num, dim3 den)
{
    return {
        (num.x + den.x - 1) / den.x,
        (num.y + den.y - 1) / den.y,
        (num.z + den.z - 1) / den.z,
    };
}

/**
 * Compute the size so each thread will compute one cell of the data.
 * Some threads may be outside the data so we need to bound check in the kernel.
 */
inline dim3 compute_grid_size(int nRows, int nCols, dim3 numThreads)
{
    return {
        ceil_div(static_cast<uint>(nRows), numThreads.x),
        ceil_div(static_cast<uint>(nCols), numThreads.y)
    };
}

// Utility to get the absolute position of the thread in the grid
// We consider the 'x' is the ROW and the 'y' is the COL because it is simpler!!!
// (it does match dim3 constructor (x, y) with convention (row, col))
#define CUGRID_ROW static_cast<int>(threadIdx.x + (blockIdx.x * blockDim.x))
#define CUGRID_COL static_cast<int>(threadIdx.y + (blockIdx.y * blockDim.y))

#define CUGRID_NUM_ROWS static_cast<int>(gridDim.x * blockDim.x)
#define CUGRID_NUM_COLS static_cast<int>(gridDim.y * blockDim.y)