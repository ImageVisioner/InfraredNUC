%%
clc;
clear;
close all;

% 读取灰度图像
image_file = '4llvip.bmp'; % 替换为你自己的图像路径
image = imread(image_file);
% load('liumat.mat')
% u=im2double(u)
% image=u
% 确保图像是灰度图像
if size(image, 3) == 3
    image = im2gray(image); % 如果是RGB图像，转换为灰度图像
end

% 参数设置
[m, n] = size(image);  % 获取图像的大小
sigma_white = 0.05;  % 白噪声的标准差
sigma_strip = 0.15;  % 条纹噪声的标准差
stripe_frequency = 120;  % 条纹的频率（每隔多少像素会有一个条纹）

% 生成条纹噪声
% 生成随机的列条纹噪声
stripe_column = rand(1, n) * sigma_strip;  % 创建一个列条纹噪声
stripe = repmat(stripe_column, m, 1);  % 将条纹扩展到整个图像高度

% 添加白噪声
white_noise = rand(m, n) * sigma_white;  % 生成白噪声

% 组合条纹噪声和白噪声
noise = stripe + white_noise;

% 可选：添加低频噪声的平滑处理（高斯滤波）
% noise = imgaussfilt(noise, 2);  % 高斯滤波器平滑噪声

% 将噪声添加到原始图像上
noisy_image = double(image) / 255 + noise;  % 将图像归一化到 [0, 1] 范围
noisy_image(noisy_image > 1) = 1;  % 限制最大值为1
noisy_image(noisy_image < 0) = 0;  % 限制最小值为0

% 显示原图和加噪后的图像
figure;
subplot(1, 2, 1);
imshow(image);
% title('Original Image');

subplot(1, 2, 2);
imshow(noisy_image);
% title('Image with Stripe Noise');

% 保存加噪后的图像
imwrite(noisy_image, 'noisy_imagesim.jpg');  % 保存噪声图像
