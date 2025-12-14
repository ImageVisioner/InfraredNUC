function divXY = divergenceXY(X, Y)
% 计算散度
divXY = [X(:, end) - X(:, 1), -diff(X, 1, 2)];
divXY = divXY + [Y(end, :) - Y(1, :); -diff(Y, 1, 1)];
end
