% 计算图像粗糙度指数
function [infraredRoughness, filteredRoughness] = calculateRoughness(infraredImage, filteredImage)
    h = fspecial('sobel');
    infraredEdge = abs(conv2(infraredImage, h, 'same'));
    filteredEdge = abs(conv2(filteredImage, h, 'same'));
    infraredEdgeT = abs(conv2(infraredImage, h', 'same'));
    filteredEdgeT = abs(conv2(filteredImage, h', 'same'));
    
    infraredEnergy = sum(infraredEdge(:)) + sum(infraredEdgeT(:));
    filteredEnergy = sum(filteredEdge(:)) + sum(filteredEdgeT(:));
    
    infraredNorm = sum(infraredImage(:));
    filteredNorm = sum(filteredImage(:));
    
    infraredRoughness = infraredEnergy / infraredNorm;
    filteredRoughness = filteredEnergy / filteredNorm;
end