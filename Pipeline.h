#pragma once

#include "cuda_compat.h"
#include "CPU.h"
#include "GPU.h"
#include <utility>

/**
 * This class just helps to disambiguate between a matrix of compile-time known and unknown size.
 * If the size is unknown, then both Rows and Cols should equals -1.
 */
template<class T, int Rows, int Cols>
struct helper_resolve_matrix_type
{
    static_assert(Rows > 0 && Cols > 0);

    template<template<class, int, int> class FixedMatrix,
             template<class> class DynamicMatrix>
    using Matrix = FixedMatrix<T, Rows, Cols>;
};

template<class T>
struct helper_resolve_matrix_type<T, -1, -1>
{
    template<template<class, int, int> class FixedMatrix,
             template<class> class DynamicMatrix>
    using Matrix = DynamicMatrix<T>;
};

/**
 * @tparam Target
 *      Should define:
 *          - Type `Matrix<T>`
 *          - Function `download(hostMatrixDst, deviceMatrixSrc)`
 *          - Function `upload(deviceMatrixDst, hostMatrixSrc)`
 *          - Function `parallel_for_each_cell(outputMatrix, ...)`
 */
template<class Target>
class Pipeline
{
public:
    /**
     * Matrix of unknown size at compile time. The data is stored on the device.
     *
     * The host can store objects of type HostMatrix but not the device.
     * The device can store objects of type DeviceMatrix but not the device.
     * But HostMatrix are convertible to DeviceMatrix when entering the device code (on the CUDA kernel launch).
     *
     * For the CPU implementation, they are just normal matrices.
     * For the GPU implementation, they are a pointer to a memory on the device (inaccessible from the host).
     */
    template<typename T> using HostMatrix = typename Target::host::template Matrix<T>;

    /**
     * Download the matrix from the device to the host.
     *
     * If the pipeline is on CPU, then just copy the pointer (no deep copy, very cheap performance cost).
     * In this case, if the output matrix is modified then the input matrix will also be modified because they
     * points to the same memory.
     */
    template<typename OutputMatrix, class T>
    static void download(OutputMatrix& outputMatrix, const HostMatrix<T>& deviceMatrixSrc)
    {
        Target::download(outputMatrix, deviceMatrixSrc);
    }

    /**
     * Upload the matrix from the host to the device.
     *
     * If the pipeline is on CPU, then only copy the pointer (no deep copy, very cheap performance cost).
     */
    template<typename InputMatrix, class T>
    static void upload(HostMatrix<T>& deviceMatrixDst, const InputMatrix& inputMatrix)
    {
        Target::upload(deviceMatrixDst, inputMatrix);
    }

    template<auto Function, class T, class... Args>
    static void parallel_for_each_cell(HostMatrix<T>& matrix, Args&&... args)
    {
        Target::template parallel_for_each_cell<Function>(matrix, std::forward<Args>(args)...);
    }
};