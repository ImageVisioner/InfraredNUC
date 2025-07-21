% 计算ICV值
function ICV = calculateICV(filteredImage)
    mu = mean(filteredImage(:));
    sigma = std(filteredImage(:));
    ICV = mu / sigma;
end