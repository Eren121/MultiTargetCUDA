#pragma once

// Make non-CUDA code able to use CUDA attributes like __host__ and __device__

#ifdef __CUDACC__
#   define CUDA_ONLY(x) x
#else
#   define CUDA_ONLY(x)
#endif

#define CUDA_DEVICE CUDA_ONLY(__device__)
#define CUDA_HOST CUDA_ONLY(__host__)
#define CUDA_BOTH CUDA_DEVICE CUDA_HOST