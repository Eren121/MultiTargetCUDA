#include "UserCode.h"

// We have to disable this warning
// That it because both the UserCode<GPU> and UserCode<CPU> functions are CUDA_BOTH
// Ideally, UserCode<GPU> should only be CUDA_DEVICE
// And UserCode<CPU> should only be CUDA_HOST
// But I don't know if its possible without ugly macros
// Since we are never calling the host version of UserCode<GPU> and device version of UserCode<CPU>, its ok.
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#   pragma nv_diagnostic push
#   pragma nv_diag_suppress 20011
#endif

template struct UserCode<GPU>;
template struct UserCode<CPU>;

template<class Target>
CUDA_BOTH void UserCode<Target>::add1ToAllElements(int iRow, int iCol, Matrix<float>& matrix)
{
    matrix(iRow, iCol) += 1.0f;
}

template<class Target>
CUDA_BOTH void UserCode<Target>::multiplyBy2AllElements(int iRow, int iCol, Matrix<float>& matrix)
{
    matrix(iRow, iCol) *= 2.0f;
}

#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#   pragma nv_diagnostic pop
#endif
