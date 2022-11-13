#pragma once

#include <opencv2/cudev/ptr2d//gpumat.hpp>
#include "Eigen.h"
#include "cuda_kernel_utils.h"

template<typename T, auto Function, typename... Args>
__global__ void callUserFunctionKernel(cv::cuda::PtrStepSz<T> matrix, const Args... args);

class GPU
{
public:
    struct host
    {
        // We use OpenCV types on the host
        // Because OpenCV automatically manage the GPU memory which is convenient

        // The OpenCV Vector will be interpreted as a Eigen Vector in the device code.
        // Because OpenCV Vector type does not work in CUDA code we have to use Eigen.
        // But there is no problem because they are memory-compatible (like just an array of float).

        template<class T>           using Matrix = cv::cudev::GpuMat_<T>;
        template<class T, int Size> using Vector = cv::Vec<T, Size>;
    };

    struct device
    {
        template<class T>           using Matrix = cv::cuda::PtrStepSz<T>;
        template<class T, int Size> using Vector = Eigen::Vector<T, Size>;
    };

public:
    template<typename OutputMatrix, class T>
    static void download(OutputMatrix& outputMatrix, const host::Matrix<T>& deviceMatrixSrc)
    {
        // Deep copy: costly!
        deviceMatrixSrc.download(outputMatrix);
    }

    template<typename InputMatrix, class T>
    static void upload(host::Matrix<T>& deviceMatrixDst, const InputMatrix& inputMatrix)
    {
        // Deep copy: costly!
        deviceMatrixDst.upload(inputMatrix);
    }

    template<auto Function, class T, class... Args>
    static void parallel_for_each_cell(host::Matrix<T>& matrix, Args&&... args)
    {
        const dim3 blockSize(16, 16);
        const dim3 gridSize = compute_grid_size(
            matrix.rows,
            matrix.cols,
            blockSize
        );

        callUserFunctionKernel<T, Function><<<gridSize, blockSize>>>(matrix, std::forward<Args>(args)...);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

template<typename T, auto Function, typename... Args>
__global__ void callUserFunctionKernel(cv::cuda::PtrStepSz<T> matrix, const Args... args)
{
    // Arguments are taken by value
    // By reference would mean copying pointer
    // But host pointers are invalid in device code

    // Iterate all pixels belonging to this thread
    for (int iRow = CUGRID_ROW; iRow < matrix.rows; iRow += CUGRID_NUM_ROWS)
    {
        for (int iCol = CUGRID_COL; iCol < matrix.cols; iCol += CUGRID_NUM_COLS)
        {
            Function(iRow, iCol, matrix, args...);
        }
    }
}