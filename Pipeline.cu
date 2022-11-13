#include "Pipeline.h"
#include "GPU.h"
#include "cuda_kernel_utils.h"

namespace gpu
{
    template<class T> using Matrix = cv::cuda::PtrStepSz<T>;

    __global__ void add1ToAllElements(Matrix<float> matrix)
    {
        for(int iRow = CUGRID_ROW; iRow < matrix.rows; iRow += CUGRID_NUM_ROWS)
        {
            for(int iCol = CUGRID_COL; iCol < matrix.cols; iCol += CUGRID_NUM_COLS)
            {
                printf("Hi from cuda from %d\n", threadIdx.x);
                matrix(iRow, iCol) += 1.0f;
            }
        }
    }
}

template<auto Function, class MatrixType>
void launchKernel(MatrixType matrix)
{
    const dim3 blockSize(16, 16);
    const dim3 gridSize = compute_grid_size(
        matrix.rows,
        matrix.cols,
        blockSize
    );

    Function<<<gridSize, blockSize>>>(matrix);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
}