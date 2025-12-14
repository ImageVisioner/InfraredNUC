#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "admm_algorithm.hpp"
#include "lra_svd_denoising.hpp"

// 计算PSNR
double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    // 检查输入有效性
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return 0.0;
    }

    cv::Mat img1_clean = img1.clone();
    cv::Mat img2_clean = img2.clone();
    cv::patchNaNs(img1_clean, 0.0);
    cv::patchNaNs(img2_clean, 0.0);

    cv::Mat diff;
    cv::absdiff(img1_clean, img2_clean, diff);
    diff = diff.mul(diff);

    cv::Scalar mse = cv::mean(diff);
    double mse_val = mse[0];  // 灰度图像只有一个通道

    if (mse_val <= 1e-10 || std::isnan(mse_val) || std::isinf(mse_val)) {
        return 100.0;  // 无限大的PSNR
    }

    double max_val = 1.0;  // 假设图像已经归一化到[0,1]
    double psnr = 10.0 * std::log10(max_val * max_val / mse_val);
    
    if (std::isnan(psnr) || std::isinf(psnr)) {
        return 0.0;
    }
    
    return psnr;
}

// 计算SSIM
double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    // 检查输入有效性
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return 0.0;
    }

    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat img1_clean = img1.clone();
    cv::Mat img2_clean = img2.clone();
    cv::patchNaNs(img1_clean, 0.0);
    cv::patchNaNs(img2_clean, 0.0);

    cv::Mat img1_2 = img1_clean.mul(img1_clean);
    cv::Mat img2_2 = img2_clean.mul(img2_clean);
    cv::Mat img1_img2 = img1_clean.mul(img2_clean);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1_clean, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_clean, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(img1_2, sigma1_2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_2, sigma2_2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img1_img2, sigma12, cv::Size(11, 11), 1.5);

    sigma1_2 -= mu1_2;
    sigma2_2 -= mu2_2;
    sigma12 -= mu1_mu2;

    cv::Mat numerator = (2 * mu1_mu2 + C1).mul(2 * sigma12 + C2);
    cv::Mat denominator = (mu1_2 + mu2_2 + C1).mul(sigma1_2 + sigma2_2 + C2);

    cv::Mat ssim_map;
    cv::divide(numerator, denominator, ssim_map);

    cv::patchNaNs(ssim_map, 0.0);

    cv::Scalar mssim = cv::mean(ssim_map);
    double ssim_val = mssim[0];  // 灰度图像只有一个通道
    
    if (std::isnan(ssim_val) || std::isinf(ssim_val)) {
        return 0.0;
    }
    
    return ssim_val;
}

int main(int argc, char* argv[]) {
    std::cout << "红外图像去条纹和降噪主程序" << std::endl;
    std::cout << "=====================================" << std::endl;

    // 检查命令行参数
    std::string inputPath = "../matlab/8.jpg";  // 默认路径
    if (argc > 1) {
        inputPath = argv[1];
    }

    // 加载测试图像
    cv::Mat originalImage = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    if (originalImage.empty()) {
        std::cerr << "无法加载图像: " << inputPath << std::endl;
        return -1;
    }

    // 转换为float并归一化
    originalImage.convertTo(originalImage, CV_32F, 1.0/255.0);

    std::cout << "图像尺寸: " << originalImage.cols << "x" << originalImage.rows << std::endl;

    // 设置ADMM算法参数
    ADMMOptions options;
    options.lambda1 = 5e-3;     // 列方向条纹正则化参数
    options.lambda2 = 1e-3;     // 行方向稀疏正则化参数
    options.lambda3 = 2e-3;     // 梯度一致性正则化参数
    options.beta1 = 0.05;       // 行方向增广拉格朗日参数
    options.beta2 = 0.1;        // 列方向增广拉格朗日参数
    options.beta3 = 0.08;       // 梯度方向增广拉格朗日参数
    options.tolerance = 1e-4;   // 收敛阈值
    options.maxIterations = 50; // 最大迭代次数

    // 执行ADMM算法进行条纹分离
    ADMMResult admmResult = admmAlgorithm(originalImage, options);

    // 计算去条纹后的图像
    cv::Mat destripedImage = originalImage - admmResult.stripeNoise;

    // 执行LRA-SVD降噪处理
    double noiseSigma = 1.0;
    srand(0);  // 设置随机种子，与Matlab保持一致
    cv::Mat finalImage = lraSvdDenoising(destripedImage, noiseSigma);

    // 显示处理结果
    std::cout << "\n=== 图像质量评估 ===" << std::endl;

    std::cout << "去条纹处理结果 (与原始图像比较):" << std::endl;
    double psnrDestripe = calculatePSNR(originalImage, destripedImage);
    double ssimDestripe = calculateSSIM(originalImage, destripedImage);
    std::cout << "PSNR: " << psnrDestripe << " dB, SSIM: " << ssimDestripe << std::endl;

    std::cout << "\n最终降噪结果 (与去条纹图像比较):" << std::endl;
    double psnrFinal = calculatePSNR(finalImage, destripedImage);
    double ssimFinal = calculateSSIM(finalImage, destripedImage);
    std::cout << "PSNR: " << psnrFinal << " dB, SSIM: " << ssimFinal << std::endl;

    // 保存处理结果
    cv::Mat destripedImage_8U, finalImage_8U, stripeNoise_8U;

    // 确保图像值在有效范围内
    cv::Mat destriped_clamped = destripedImage.clone();
    cv::Mat final_clamped = finalImage.clone();
    cv::max(destriped_clamped, 0.0, destriped_clamped);
    cv::min(destriped_clamped, 1.0, destriped_clamped);
    cv::max(final_clamped, 0.0, final_clamped);
    cv::min(final_clamped, 1.0, final_clamped);

    destriped_clamped.convertTo(destripedImage_8U, CV_8U, 255.0);
    final_clamped.convertTo(finalImage_8U, CV_8U, 255.0);
    admmResult.stripeNoise.convertTo(stripeNoise_8U, CV_8U, 255.0);

    cv::imwrite("destriped_image.png", destripedImage_8U);
    cv::imwrite("final_denoised_image.png", finalImage_8U);

    std::cout << "\n处理结果已保存为 destriped_image.png 和 final_denoised_image.png" << std::endl;

    // 可视化结果（如果有GUI支持）
    if (cv::getWindowProperty("Original", cv::WND_PROP_VISIBLE) < 0) {
        cv::namedWindow("Original", cv::WINDOW_NORMAL);
        cv::namedWindow("Destriped", cv::WINDOW_NORMAL);
        cv::namedWindow("Stripe Noise", cv::WINDOW_NORMAL);
        cv::namedWindow("Final Denoised", cv::WINDOW_NORMAL);

        cv::Mat originalDisplay, destripedDisplay, stripeDisplay, finalDisplay;
        originalImage.convertTo(originalDisplay, CV_8U, 255.0);
        destripedImage.convertTo(destripedDisplay, CV_8U, 255.0);
        admmResult.stripeNoise.convertTo(stripeDisplay, CV_8U, 255.0);
        finalImage.convertTo(finalDisplay, CV_8U, 255.0);

        cv::imshow("Original", originalDisplay);
        cv::imshow("Destriped", destripedDisplay);
        cv::imshow("Stripe Noise", stripeDisplay);
        cv::imshow("Final Denoised", finalDisplay);

        cv::waitKey(0);
    }

    return 0;
}
