#pragma once

#include <opencv2/opencv.hpp>

// LRA-SVD降噪算法
cv::Mat lraSvdDenoising(const cv::Mat& inputImage, double noiseSigma);
