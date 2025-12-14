#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// ADMM算法参数结构体
struct ADMMOptions {
    double lambda1 = 5e-3;     // 列方向条纹正则化参数
    double lambda2 = 1e-3;     // 行方向稀疏正则化参数
    double lambda3 = 2e-3;     // 梯度一致性正则化参数
    double beta1 = 0.05;       // 行方向增广拉格朗日参数
    double beta2 = 0.1;        // 列方向增广拉格朗日参数
    double beta3 = 0.08;       // 梯度方向增广拉格朗日参数
    double tolerance = 1e-4;   // 收敛阈值
    int maxIterations = 50;    // 最大迭代次数
};

// ADMM算法结果结构体
struct ADMMResult {
    cv::Mat stripeNoise;
    int iterationCount;
    std::vector<double> relChanges;
};

// ADMM算法主函数
ADMMResult admmAlgorithm(const cv::Mat& inputImage, const ADMMOptions& options);
