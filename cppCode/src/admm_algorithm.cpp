#include "admm_algorithm.hpp"
#include "image_processing.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// 符号函数
cv::Mat sign(const cv::Mat& x) {
    cv::Mat result = cv::Mat::zeros(x.size(), x.type());
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            float val = x.at<float>(i, j);
            if (val > 0) result.at<float>(i, j) = 1.0f;
            else if (val < 0) result.at<float>(i, j) = -1.0f;
            else result.at<float>(i, j) = 0.0f;
        }
    }
    return result;
}

// L2,1范数
float normL21(const cv::Mat& x, int col) {
    cv::Mat column = x.col(col);
    float norm = 0.0f;
    for (int i = 0; i < column.rows; i++) {
        norm += column.at<float>(i) * column.at<float>(i);
    }
    return std::sqrt(norm);
}

// ADMM算法主函数
ADMMResult admmAlgorithm(const cv::Mat& inputImage, const ADMMOptions& options) {
    std::cout << "开始ADMM算法进行条纹分离..." << std::endl;

    ADMMResult result;
    cv::Size imageSize = inputImage.size();

    // 计算频率域系数
    FreqCoeff freqCoeff = getFreqCoeff(inputImage);

    // 定义算子
    auto gradX = forwardDiffX;
    auto gradY = forwardDiffY;
    auto divergence = divergenceXY;

    // 初始化ADMM变量
    cv::Mat originalImage = inputImage.clone();
    cv::Mat stripeNoise = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat gradVarX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat sparseVarY = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat consistencyVarZ = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat dualVarP1 = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat dualVarP2 = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat dualVarP3 = cv::Mat::zeros(imageSize, CV_32F);

    // 提取算法参数
    double lambda1 = options.lambda1;
    double lambda2 = options.lambda2;
    double lambda3 = options.lambda3;
    double beta1 = options.beta1;
    double beta2 = options.beta2;
    double beta3 = options.beta3;
    double tolerance = options.tolerance;
    int maxIterations = options.maxIterations;

    // 构建线性系统系数矩阵（频域）
    cv::Mat denominator = beta3 * freqCoeff.eigsGradXtGradX +
                         cv::Mat::ones(imageSize, CV_32F) * beta2 +
                         beta1 * freqCoeff.eigsGradYtGradY;

    // 初始化迭代控制变量
    int iterationCount = 0;
    double relChange = 1.0;
    cv::Mat prevCleanImage;

    // 主迭代循环
    while (relChange > tolerance && iterationCount < maxIterations) {
        // x子问题：L1正则化（行方向梯度）
        cv::Mat tempV1 = gradY(stripeNoise) + dualVarP1 / beta1;
        cv::Mat tempV1_abs;
        cv::absdiff(tempV1, cv::Scalar(0), tempV1_abs);
        cv::Mat threshold_mask = tempV1_abs > lambda2 / beta1;
        gradVarX = sign(tempV1).mul(cv::max(0.0, tempV1_abs - lambda2 / beta1));

        // y子问题：L2,1正则化（列方向稀疏）
        for (int col = 0; col < imageSize.width; col++) {
            cv::Mat tempR = stripeNoise.col(col) + dualVarP2.col(col) / beta2;
            float norm_val = normL21(tempR, 0);
            if (norm_val > lambda1 / beta2) {
                sparseVarY.col(col) = tempR * (1.0f - lambda1 / (beta2 * norm_val));
            } else {
                sparseVarY.col(col) = cv::Mat::zeros(tempR.size(), CV_32F);
            }
        }

        // z子问题：L1正则化（梯度一致性）
        cv::Mat tempV2 = gradX(originalImage - stripeNoise) + dualVarP3 / beta3;
        cv::Mat tempV2_abs;
        cv::absdiff(tempV2, cv::Scalar(0), tempV2_abs);
        consistencyVarZ = sign(tempV2).mul(cv::max(0.0, tempV2_abs - lambda3 / beta3));

        // s子问题：线性系统求解（频域）
        cv::Mat temp1 = beta3 * gradX(originalImage) - beta3 * consistencyVarZ + dualVarP3;
        cv::Mat temp2 = beta2 * sparseVarY - dualVarP2;
        cv::Mat temp3 = beta1 * gradVarX - dualVarP1;
        cv::Mat tempS1 = divergence(temp1, temp3) + temp2;

        // 频域求解
        cv::Mat tempS1_fft, denominator_fft;
        cv::dft(tempS1, tempS1_fft, cv::DFT_COMPLEX_OUTPUT);
        cv::dft(denominator, denominator_fft, cv::DFT_COMPLEX_OUTPUT);

        // 避免除零
        cv::Mat denominator_safe = denominator + 1e-10;

        cv::Mat tempS2_fft;
        std::vector<cv::Mat> planes_tempS1(2), planes_den(2), planes_result(2);
        cv::split(tempS1_fft, planes_tempS1);
        cv::split(denominator_fft, planes_den);

        for (int i = 0; i < 2; i++) {
            planes_result[i] = planes_tempS1[i].mul(1.0 / (planes_den[i] + 1e-10));
        }

        cv::merge(planes_result, tempS2_fft);
        cv::dft(tempS2_fft, stripeNoise, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        // 更新对偶变量
        dualVarP1 = dualVarP1 + beta1 * (gradY(stripeNoise) - gradVarX);
        dualVarP2 = dualVarP2 + beta2 * (stripeNoise - sparseVarY);
        dualVarP3 = dualVarP3 + beta3 * (gradX(originalImage - stripeNoise) - consistencyVarZ);

        // 计算收敛指标
        cv::Mat cleanImage = originalImage - stripeNoise;
        if (iterationCount > 0) {
            cv::Mat diff = cleanImage - prevCleanImage;
            double fro_norm_diff = cv::norm(diff, cv::NORM_L2);
            double fro_norm_clean = cv::norm(cleanImage, cv::NORM_L2);
            relChange = fro_norm_diff / (fro_norm_clean + 1e-10);
        }
        result.relChanges.push_back(relChange);
        iterationCount++;

        // 保存上一次结果用于收敛判断
        prevCleanImage = cleanImage.clone();

        // 显示迭代进度
        if (iterationCount % 10 == 0) {
            std::cout << "ADMM迭代 " << iterationCount << "/" << maxIterations
                     << ", 相对变化: " << relChange << std::endl;
        }
    }

    result.stripeNoise = stripeNoise;
    result.iterationCount = iterationCount;

    std::cout << "ADMM算法完成，共" << iterationCount << "次迭代" << std::endl;

    return result;
}
