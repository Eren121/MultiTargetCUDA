#pragma once

#include <opencv2/opencv.hpp>

struct CPU
{
    struct host
    {
        template<class T>           using Matrix = cv::Mat_<T>;
        template<class T, int Size> using Vector = cv::Vec<T, Size>;
    };

    // On CPU, both GPU and CPU types are the same
    struct device : host {};

    template<class T>
    static void download(host::Matrix<T>& hostMatrixDst, const device::Matrix<T>& deviceMatrixSrc)
    {
        // Shallow copy
        hostMatrixDst = deviceMatrixSrc;
    }

    template<class T>
    static void upload(device::Matrix<T>& deviceMatrixDst, const host::Matrix<T>& hostMatrixSrc)
    {
        // Shallow copy
        deviceMatrixDst = hostMatrixSrc;
    }

    template<auto Function, class T, class... Args>
    static void parallel_for_each_cell(host::Matrix<T>& matrix, Args&&... args)
    {
        for(int iRow = 0; iRow < matrix.rows; iRow++)
        {
            for(int iCol = 0; iCol < matrix.cols; iCol++)
            {
                Function(iRow, iCol, matrix, std::forward<Args>(args)...);
            }
        }
    }
};