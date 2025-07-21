% 计算AVGE值
function AVGE = calculateAVGE(infraredImage, filteredImage)
    [gradX_ir, gradY_ir] = gradient(infraredImage);
    [gradX_filt, gradY_filt] = gradient(filteredImage);
    gradDifference = abs(gradX_ir - gradX_filt) + abs(gradY_ir - gradY_filt);
    AVGE = mean(gradDifference(:));
end