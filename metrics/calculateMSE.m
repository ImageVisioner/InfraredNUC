% 计算MSE值
function mseValue = calculateMSE(infraredImage, filteredImage)
    mseValue = mean((infraredImage - filteredImage).^2, 'all');
end