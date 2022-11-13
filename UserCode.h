#pragma once

#include "Pipeline.h"

template<class Target>
class UserCode
{
public:
    template<class T>           using Matrix = typename Target::device::template Matrix<T>;
    template<class T, int Size> using Vector = typename Target::device::template Vector<T, Size>;

    CUDA_BOTH
    static void add1ToAllElements(int iRow, int iCol, Matrix<float>& matrix);

    CUDA_BOTH
    static void multiplyBy2AllElements(int iRow, int iCol, Matrix<float>& matrix);
};