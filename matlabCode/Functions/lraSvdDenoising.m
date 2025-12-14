function imOut = lraSvdDenoising(inputImage, noiseSigma)
% LRA-SVD图像降噪算法

if noiseSigma <= 15
    blockSize = 3;
    numBlocks = 85;
    weight = 0.5;
    stepSize = [1, 0.5];
    numIterations = 2;
    lambda = 0.65;
elseif noiseSigma <= 30
    blockSize = 5;
    numBlocks = 85;
    weight = 0.5;
    stepSize = [1, 0.5];
    numIterations = 2;
    lambda = 0.65;
elseif noiseSigma <= 50
    blockSize = 7;
    numBlocks = 85;
    weight = 0.5;
    stepSize = [1, 0.5];
    numIterations = 2;
    lambda = 0.63;
else
    blockSize = 9;
    numBlocks = 75;
    weight = 0.5;
    stepSize = [1, 0.6, 0.2];
    numIterations = 3;
    lambda = 0.63;
end

step = min(6, blockSize-1);
currentSigma = noiseSigma;
inputImageRef = inputImage;
blockSizeRef = blockSize;

[imageHeight, imageWidth, ~] = size(inputImageRef);
numRows = imageHeight - blockSizeRef + 1;
numCols = imageWidth - blockSizeRef + 1;
rowIndices = [1:numRows];
colIndices = [1:numCols];

imOut = inputImageRef;
noiseEstimate = currentSigma;

fprintf('开始LRA-SVD降噪，共%d次迭代...\n', numIterations);

for iteration = 1:numIterations
    imOut = imOut + stepSize(iteration) * (inputImageRef - imOut);
    diff = imOut - inputImageRef;

    varianceDiff = noiseEstimate^2 - mean(mean(diff.^2));
    if iteration == 1
        currentSigma = sqrt(abs(varianceDiff));
    else
        currentSigma = sqrt(abs(varianceDiff)) * lambda;
    end

    blockArray = blockMatching(imOut, blockSize, step, numBlocks);
    patchMatrix = im2Patch(imOut, blockSize);
    denoisedPatches = zeros(size(patchMatrix));
    weightMatrix = zeros(size(patchMatrix));
    numGroups = size(blockArray, 2);

    for groupIdx = 1:numGroups
        blockGroup = patchMatrix(:, blockArray(:, groupIdx));
        [denoisedPatches(:, blockArray(:, groupIdx)), weightMatrix(:, blockArray(:, groupIdx)), ~] = lowRankSvdEstimation(double(blockGroup), currentSigma, groupIdx);
    end

    imOut = zeros(imageHeight, imageWidth);
    weightAccum = zeros(imageHeight, imageWidth);
    pixelIdx = 0;

    for i = 1:blockSize
        for j = 1:blockSize
            pixelIdx = pixelIdx + 1;
            imOut(rowIndices-1+i, colIndices-1+j) = imOut(rowIndices-1+i, colIndices-1+j) + reshape(denoisedPatches(pixelIdx, :)', [numRows, numCols]);
            weightAccum(rowIndices-1+i, colIndices-1+j) = weightAccum(rowIndices-1+i, colIndices-1+j) + reshape(weightMatrix(pixelIdx, :)', [numRows, numCols]);
        end
    end

    imOut = imOut ./ (weightAccum + eps);
    fprintf('LRA-SVD迭代 %d/%d 完成\n', iteration, numIterations);
end

fprintf('LRA-SVD降噪完成\n');


function [X, W, r] = lowRankSvdEstimation(Y, nsig, i)
[U0, S0, V0] = svd(full(Y), 'econ');
S0 = diag(S0);
S2 = S0.^2;
[ms, ns] = size(Y);
num = size(S2, 1);
ss = 0;

for j = num:-1:1
    ss = ss + S2(j);
    sk = ss / (ms * ns);
    if sk > nsig.^2
        r = j + 1;
        break;
    else
        r = 1;
    end
end

U = U0(:, 1:r);
V = V0(:, 1:r);
X = U * diag(S0(1:r)) * V';

if r == size(Y, 1)
    wei = 1 / size(Y, 1);
else
    wei = (size(Y, 1) - r) / size(Y, 1);
end

W = wei * ones(size(X));
X = X * wei;

function X = im2Patch(im, win)
f = win;
N = size(im, 1) - f + 1;
M = size(im, 2) - f + 1;
L = N * M;
X = zeros(f*f, L, 'single');
k = 0;
for i = 1:f
    for j = 1:f
        k = k + 1;
        blk = im(i:end-f+i, j:end-f+j);
        X(k, :) = blk(:)';
    end
end

function posArr = blockMatching(im, win, step, nblk)
S = 21;
f = win;
f2 = f^2;
s = step;

N = size(im, 1) - f + 1;
M = size(im, 2) - f + 1;

r = [1:s:N];
r = [r, r(end)+1:N];
c = [1:s:M];
c = [c, c(end)+1:M];

L = N * M;
X = zeros(f*f, L, 'single');
k = 0;
for i = 1:f
    for j = 1:f
        k = k + 1;
        blk = im(i:end-f+i, j:end-f+j);
        X(k, :) = blk(:)';
    end
end

I = (1:L);
I = reshape(I, N, M);

N1 = length(r);
M1 = length(c);
posArr = zeros(nblk, N1*M1);
X = X';

for i = 1:N1
    for j = 1:M1
        row = r(i);
        col = c(j);
        off = (col-1)*N + row;
        off1 = (j-1)*N1 + i;
        rmin = max(row-S, 1);
        rmax = min(row+S, N);
        cmin = max(col-S, 1);
        cmax = min(col+S, M);
        idx = I(rmin:rmax, cmin:cmax);
        idx = idx(:);
        B = X(idx, :);
        v = X(off, :);
        dis = (B(:, 1) - v(1)).^2;
        for k = 2:f2
            dis = dis + (B(:, k) - v(k)).^2;
        end
        [~, ind] = sort(dis);
        posArr(:, off1) = idx(ind(1:nblk));
    end
end


