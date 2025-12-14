function freqCoeff = getFreqCoeff(inputImage)
% 计算频率域卷积系数
%
% 输入参数：
%   inputImage: 输入图像
%
% 输出参数：
%   freqCoeff: 频率域系数结构体

imageSize = size(inputImage);
freqCoeff.eigsGradXtGradX = abs(psf2otf([1, -1], imageSize)).^2;
freqCoeff.eigsGradYtGradY = abs(psf2otf([1; -1], imageSize)).^2;
end
