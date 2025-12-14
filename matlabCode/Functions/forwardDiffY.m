function gradY = forwardDiffY(image)
% y方向的一阶向前有限差分
gradY = [diff(image, 1, 1); image(1, :) - image(end, :)];
end
