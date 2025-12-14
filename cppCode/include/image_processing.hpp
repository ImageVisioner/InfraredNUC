#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// 差分运算
cv::Mat forwardDiffX(const cv::Mat& image);
cv::Mat forwardDiffY(const cv::Mat& image);

// 散度运算
cv::Mat divergenceXY(const cv::Mat& X, const cv::Mat& Y);

// 频域系数结构体
struct FreqCoeff {
    cv::Mat eigsGradXtGradX;
    cv::Mat eigsGradYtGradY;
};

// 计算频域系数
FreqCoeff getFreqCoeff(const cv::Mat& inputImage);

// 图像块转换为矩阵
cv::Mat im2Patch(const cv::Mat& im, int win);

// 块匹配
cv::Mat blockMatching(const cv::Mat& im, int win, int step, int nblk);

// 低秩SVD估计
void lowRankSvdEstimation(const cv::Mat& Y, double nsig, cv::Mat& X, cv::Mat& W, int& r);
