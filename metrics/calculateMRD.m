% 计算MRD值
function MRD = calculateMRD(infraredImage, filteredImage)
    epsilon = 1e-5;
    MRD = mean(mean(abs((infraredImage - filteredImage) ./ (infraredImage + epsilon))));
end