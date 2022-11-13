cmake_minimum_required(VERSION 3.6)
project(test LANGUAGES CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

# Allow to include include <cuda_runtime.h> from normal C++ file.
include_directories("${CUDA_INCLUDE_DIRS}")

add_executable(test main.cu Pipeline.h CPU.h GPU.h Eigen.h cuda_compat.h cuda_kernel_utils.h UserCode.h UserCode.cu)
target_link_libraries(test ${OpenCV_LIBRARIES} Eigen3::Eigen)

set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)