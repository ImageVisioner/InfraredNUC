% 计算SSIM值
function ssimValue = calculateSSIM(infraredImage, filteredImage)
    ssimValue = ssim(uint8(infraredImage), uint8(filteredImage));
end