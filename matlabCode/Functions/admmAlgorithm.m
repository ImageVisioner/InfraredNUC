function [stripeNoise, iterationCount, relChanges] = admmAlgorithm(inputImage, options)
% ADMM算法主函数 - 红外图像条纹噪声分离
%
% 输入参数：
%   inputImage: 输入图像
%   options: 参数结构体
% 输出参数：
%   stripeNoise: 分离出的条纹噪声
%   iterationCount: 实际迭代次数
%   relChanges: 相对变化记录

fprintf('开始ADMM算法进行条纹分离...\n');
[imageHeight, imageWidth] = size(inputImage);
freqCoeff = getFreqCoeff(inputImage);    % 计算频率域系数
gradX = defGradX;                        % x方向梯度算子
gradY = defGradY;                        % y方向梯度算子
divergence = defDivergence;              % 散度算子

% 初始化ADMM变量
originalImage = inputImage;              % 原始图像
stripeNoise = zeros(imageHeight, imageWidth);    % 条纹分量初始化
gradVarX = zeros(imageHeight, imageWidth);       % x方向梯度变量
sparseVarY = zeros(imageHeight, imageWidth);     % y方向稀疏变量
consistencyVarZ = zeros(imageHeight, imageWidth); % 梯度一致性变量
dualVarP1 = zeros(imageHeight, imageWidth);      % 对偶变量1
dualVarP2 = zeros(imageHeight, imageWidth);      % 对偶变量2
dualVarP3 = zeros(imageHeight, imageWidth);      % 对偶变量3

% 提取算法参数
lambda1 = options.lambda1;
lambda2 = options.lambda2;
lambda3 = options.lambda3;
beta1 = options.beta1;
beta2 = options.beta2;
beta3 = options.beta3;
tolerance = options.tolerance;
maxIterations = options.maxIterations;

% 构建线性系统系数矩阵（频域）
denominator = beta3 * freqCoeff.eigsGradXtGradX + beta2 * ones(imageHeight, imageWidth) + beta1 * freqCoeff.eigsGradYtGradY;

% 初始化迭代控制变量
iterationCount = 0;      % 迭代计数器
relChange = 1;           % 相对变化量
relChanges = [];         % 记录每次迭代的变化

% 主迭代循环
while relChange > tolerance && iterationCount < maxIterations
    % x子问题：L1正则化（行方向梯度）
    tempV1 = gradY(stripeNoise) + dualVarP1 / beta1;
    gradVarX = sign(tempV1) .* max(0, abs(tempV1) - lambda2 / beta1);

    % y子问题：L2,1正则化（列方向稀疏）
    for col = 1:imageWidth
        tempR = stripeNoise(:, col) + dualVarP2(:, col) / beta2;
        sparseVarY(:, col) = tempR .* max(norm(tempR) - lambda1 / beta2, 0) / (norm(tempR) + eps);
    end

    % z子问题：L1正则化（梯度一致性）
    tempV2 = gradX(originalImage - stripeNoise) + dualVarP3 / beta3;
    consistencyVarZ = sign(tempV2) .* max(0, abs(tempV2) - lambda3 / beta3);

    % s子问题：线性系统求解（频域）
    temp1 = beta3 * gradX(originalImage) - beta3 * consistencyVarZ + dualVarP3;
    temp2 = beta2 * sparseVarY - dualVarP2;
    temp3 = beta1 * gradVarX - dualVarP1;
    tempS1 = divergence(temp1, temp3) + temp2;
    tempS2 = fft2(tempS1) ./ (denominator + eps);
    stripeNoise = real(ifft2(tempS2));

    % 更新对偶变量
    dualVarP1 = dualVarP1 + beta1 * (gradY(stripeNoise) - gradVarX);
    dualVarP2 = dualVarP2 + beta2 * (stripeNoise - sparseVarY);
    dualVarP3 = dualVarP3 + beta3 * (gradX(originalImage - stripeNoise) - consistencyVarZ);

    % 计算收敛指标
    cleanImage = originalImage - stripeNoise;
    if iterationCount > 0
        relChange = norm(cleanImage - prevCleanImage, 'fro') / norm(cleanImage, 'fro');
    end
    relChanges = [relChanges, relChange];
    iterationCount = iterationCount + 1;

    % 保存上一次结果用于收敛判断
    prevCleanImage = cleanImage;

    % 显示迭代进度
    if mod(iterationCount, 10) == 0
        fprintf('ADMM迭代 %d/%d, 相对变化: %.6f\n', iterationCount, maxIterations, relChange);
    end
end

fprintf('ADMM算法完成，共%d次迭代\n', iterationCount);
end
