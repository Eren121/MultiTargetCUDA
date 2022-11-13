#pragma once

// Supress Eigen annoying warnings

#ifdef __GNUC__
// Compiler is GCC
#   pragma GCC system_header
#elif defined(_MSV_VER)
// Compiler is Microsoft Visual Studio
#   pragma warning(push, 0)
#endif

#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
// System header is not enough, we have still to disable some CUDA warnings
    
    // CUDA source file with pragma support (>11.5)
#   pragma nv_diagnostic push

    // __host__ / __device__ annotation is ignored on a function that is explicitly defaulted on its first declaration
#   pragma nv_diag_suppress 20012
#endif

#include <Eigen/Dense>

#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#   pragma nv_diagnostic pop
#endif

#if defined(_MSV_VER)
#   pragma warning(pop)
#endif