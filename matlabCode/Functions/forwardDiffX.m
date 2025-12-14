function gradX = forwardDiffX(image)
% x方向的一阶向前有限差分
gradX = [diff(image, 1, 2), image(:, 1) - image(:, end)];
end
