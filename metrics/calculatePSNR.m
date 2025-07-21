% 计算PSNR值
function psnrValue = calculatePSNR(mseValue)
    maxPixelValue = 255.0;
    psnrValue = 20 * log10(maxPixelValue / sqrt(mseValue));
end