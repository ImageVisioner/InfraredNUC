#include "lra_svd_denoising.hpp"
#include "image_processing.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// LRA-SVD降噪算法
cv::Mat lraSvdDenoising(const cv::Mat& inputImage, double noiseSigma) {
    int blockSize, numBlocks;
    double lambda;
    std::vector<double> stepSize;
    int numIterations;

    // 根据噪声水平设置参数
    if (noiseSigma <= 15) {
        blockSize = 3;
        numBlocks = 85;
        stepSize = {1.0, 0.5};
        numIterations = 2;
        lambda = 0.65;
    } else if (noiseSigma <= 30) {
        blockSize = 5;
        numBlocks = 85;
        stepSize = {1.0, 0.5};
        numIterations = 2;
        lambda = 0.65;
    } else if (noiseSigma <= 50) {
        blockSize = 7;
        numBlocks = 85;
        stepSize = {1.0, 0.5};
        numIterations = 2;
        lambda = 0.63;
    } else {
        blockSize = 9;
        numBlocks = 75;
        stepSize = {1.0, 0.6, 0.2};
        numIterations = 3;
        lambda = 0.63;
    }

    int step = std::min(6, blockSize - 1);
    double currentSigma = noiseSigma;

    cv::Mat imOut = inputImage.clone();
    cv::Mat inputImageRef = inputImage.clone();
    int blockSizeRef = blockSize;

    cv::Size imageSize = inputImageRef.size();
    int numRows = imageSize.height - blockSizeRef + 1;
    int numCols = imageSize.width - blockSizeRef + 1;

    std::vector<int> rowIndices(numRows), colIndices(numCols);
    for (int i = 0; i < numRows; i++) rowIndices[i] = i;
    for (int i = 0; i < numCols; i++) colIndices[i] = i;

    imOut = inputImageRef.clone();
    double noiseEstimate = currentSigma;

    std::cout << "开始LRA-SVD降噪，共" << numIterations << "次迭代..." << std::endl;

    for (int iteration = 0; iteration < numIterations; iteration++) {
        // 检查当前迭代的输入是否有问题
        cv::Mat imOut_check = imOut.clone();
        cv::patchNaNs(imOut_check, 0.0);

        imOut_check = imOut_check + stepSize[iteration] * (inputImageRef - imOut_check);
        cv::patchNaNs(imOut_check, 0.0);
        imOut = imOut_check;

        cv::Mat diff = imOut - inputImageRef;

        cv::Scalar mean_diff, std_diff;
        cv::meanStdDev(diff, mean_diff, std_diff);
        double varianceDiff = std_diff[0] * std_diff[0] - noiseEstimate * noiseEstimate;

        if (iteration == 0) {
            currentSigma = std::sqrt(std::abs(varianceDiff));
        } else {
            currentSigma = std::sqrt(std::abs(varianceDiff)) * lambda;
        }

        // 块匹配
        cv::Mat blockArray = blockMatching(imOut, blockSize, step, numBlocks);
        cv::Mat patchMatrix = im2Patch(imOut, blockSize);

        // 检查patchMatrix是否有nan
        cv::patchNaNs(patchMatrix, 0.0);

        cv::Mat denoisedPatches = cv::Mat::zeros(patchMatrix.rows, patchMatrix.cols, patchMatrix.type());
        cv::Mat weightMatrix = cv::Mat::zeros(patchMatrix.rows, patchMatrix.cols, patchMatrix.type());
        int numGroups = blockArray.cols;

        for (int groupIdx = 0; groupIdx < numGroups; groupIdx++) {
            cv::Mat blockGroup_indices = blockArray.col(groupIdx);
            cv::Mat blockGroup(patchMatrix.rows, blockGroup_indices.rows, patchMatrix.type());

            for (int i = 0; i < blockGroup_indices.rows; i++) {
                int idx = blockGroup_indices.at<int>(i);
                // 边界检查
                if (idx >= 0 && idx < patchMatrix.cols) {
                    patchMatrix.col(idx).copyTo(blockGroup.col(i));
                } else {
                    // 如果索引越界，使用零填充
                    blockGroup.col(i).setTo(0);
                }
            }

            cv::Mat X, W;
            int r;
            lowRankSvdEstimation(blockGroup, currentSigma, X, W, r);

            for (int i = 0; i < blockGroup_indices.rows; i++) {
                int idx = blockGroup_indices.at<int>(i);
                // 边界检查
                if (idx >= 0 && idx < denoisedPatches.cols) {
                    X.col(i).copyTo(denoisedPatches.col(idx));
                    W.col(i).copyTo(weightMatrix.col(idx));
                }
            }
        }

        // 重建图像
        cv::Mat newImOut = cv::Mat::zeros(imageSize, imOut.type());
        cv::Mat weightAccum = cv::Mat::zeros(imageSize, imOut.type());

        int pixelIdx = 0;
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                pixelIdx++;

                // 从denoisedPatches和weightMatrix中提取当前像素位置的patch数据
                // pixelIdx对应块内的像素位置，需要重建为numRows x numCols的patch
                cv::Mat denoised_patch = cv::Mat::zeros(numRows, numCols, imOut.type());
                cv::Mat weight_patch = cv::Mat::zeros(numRows, numCols, imOut.type());

                // 将列向量数据重塑为patch矩阵（列优先顺序）
                for (int r = 0; r < numRows; r++) {
                    for (int c = 0; c < numCols; c++) {
                        int idx = r + c * numRows;  // 与im2Patch中的存储顺序一致（列优先）
                        if (idx >= 0 && idx < denoisedPatches.cols && pixelIdx - 1 < denoisedPatches.rows) {
                            denoised_patch.at<float>(r, c) = denoisedPatches.at<float>(pixelIdx - 1, idx);
                            weight_patch.at<float>(r, c) = weightMatrix.at<float>(pixelIdx - 1, idx);
                        }
                    }
                }

                // 将patch累加到图像的所有可能位置
                // 对于每个块位置(rowIdx, colIdx)，patch应该放在(rowIdx+i, colIdx+j)
                for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    for (int colIdx = 0; colIdx < numCols; colIdx++) {
                        int imgRow = rowIdx + i;
                        int imgCol = colIdx + j;
                        
                        // 边界检查
                        if (imgRow >= 0 && imgRow < imageSize.height && 
                            imgCol >= 0 && imgCol < imageSize.width) {
                            newImOut.at<float>(imgRow, imgCol) += denoised_patch.at<float>(rowIdx, colIdx);
                            weightAccum.at<float>(imgRow, imgCol) += weight_patch.at<float>(rowIdx, colIdx);
                        }
                    }
                }
            }
        }

        // 避免除零
        cv::Mat weightAccum_safe = weightAccum + 1e-10;
        cv::divide(newImOut, weightAccum_safe, newImOut);

        // 确保结果不包含nan
        cv::patchNaNs(newImOut, 0.0);
        imOut = newImOut.clone();

        std::cout << "LRA-SVD迭代 " << (iteration + 1) << "/" << numIterations << " 完成" << std::endl;
    }

    std::cout << "LRA-SVD降噪完成" << std::endl;

    return imOut;
}
