#include "Pipeline.h"
#include "CPU.h"
#include "GPU.h"
#include "UserCode.h"
#include <chrono>
#include <iostream>
#include <iomanip>

template<typename Target>
void profile(cv::Mat_<float>& toModify, const char* title)
{
    using namespace std::chrono;
    const auto start = high_resolution_clock::now();

    {
        using Pipeline = Pipeline<Target>;
        using UserCode = UserCode<Target>;

        typename Target::host::template Matrix<float> d_toModify;
        Pipeline::upload(d_toModify, toModify);

        Pipeline::template parallel_for_each_cell<UserCode::add1ToAllElements>(d_toModify);
        Pipeline::template parallel_for_each_cell<UserCode::multiplyBy2AllElements>(d_toModify);

        cv::Mat_<float> res;
        Pipeline::download(toModify, d_toModify);
    }

    const auto end = high_resolution_clock::now();
    const float elapsed = duration_cast<std::chrono::duration<float>>(end - start).count();
    std::cout << "Elapsed for " << title << ": " << std::fixed << std::setprecision(2)  << elapsed << std::endl;
};

int main()
{
    const int rows = 10000;
    const int cols = 10000;

    cv::Mat_<float> cpuMat(rows, cols, 0.0f);
    cv::Mat_<float> gpuMat(rows, cols, 0.0f);

    profile<CPU>(cpuMat, "CPU");
    profile<GPU>(gpuMat, "GPU");

    std::cout << "diff = " << cv::norm(cpuMat - gpuMat) << std::endl;
}