%% 红外图像去条纹和降噪主程序
%% 作者：chenhua liu
clc;
clear;
close all;

% 添加函数路径
addpath('./Functions');

% 加载测试图像
originalImage = imread('8.jpg');
originalImage = im2double(originalImage);
originalImage = im2gray(originalImage);

% 设置ADMM算法参数
options.lambda1 = 5e-3;     % 列方向条纹正则化参数
options.lambda2 = 1e-3;     % 行方向稀疏正则化参数
options.lambda3 = 2e-3;     % 梯度一致性正则化参数
options.beta1 = 0.05;       % 行方向增广拉格朗日参数
options.beta2 = 0.1;        % 列方向增广拉格朗日参数
options.beta3 = 0.08;       % 梯度方向增广拉格朗日参数
options.tolerance = 1e-4;   % 收敛阈值
options.maxIterations = 50; % 最大迭代次数

% 执行ADMM算法进行条纹分离
[stripeNoise, iterationCount, relChanges] = admmAlgorithm(originalImage, options);

% 计算去条纹后的图像
destripedImage = originalImage - stripeNoise;

% 执行LRA-SVD降噪处理
noiseSigma = 1.0;
randn('seed', 0);
finalImage = lraSvdDenoising(destripedImage, noiseSigma);

% 显示处理结果
figure;
subplot(2,3,1), imshow(originalImage, []), title('原始条纹图像')
subplot(2,3,2), imshow(destripedImage, []), title('去条纹后图像')
subplot(2,3,3), imshow(stripeNoise, []), title('估计的条纹噪声')
subplot(2,3,4), imshow(finalImage, []), title('最终降噪结果')

% 计算并显示残留噪声
residualNoise = destripedImage - finalImage;
subplot(2,3,5), imshow(residualNoise, []), title('残留噪声')

% 计算图像质量指标
fprintf('\n=== 图像质量评估 ===\n');

fprintf('去条纹处理结果 (与原始图像比较):\n');
psnrDestripe = psnr(originalImage, destripedImage);
ssimDestripe = ssim(originalImage, destripedImage);
fprintf('PSNR: %.4f dB, SSIM: %.4f\n', psnrDestripe, ssimDestripe);

fprintf('\n最终降噪结果 (与去条纹图像比较):\n');
psnrFinal = psnr(finalImage, destripedImage);
ssimFinal = ssim(finalImage, destripedImage);
fprintf('PSNR: %.4f dB, SSIM: %.4f\n', psnrFinal, ssimFinal);

% 保存处理结果
imwrite(uint8(destripedImage * 255), 'destriped_image.png');
imwrite(uint8(finalImage * 255), 'final_denoised_image.png');
fprintf('\n处理结果已保存为 destriped_image.png 和 final_denoised_image.png\n');