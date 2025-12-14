#include "image_processing.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>

// x方向的一阶向前有限差分
cv::Mat forwardDiffX(const cv::Mat& image) {
    CV_Assert(image.type() == CV_32F || image.type() == CV_64F);

    cv::Mat gradX = cv::Mat::zeros(image.size(), image.type());

    // 计算差分：gradX(:, 1:end-1) = image(:, 2:end) - image(:, 1:end-1)
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols - 1; j++) {
            gradX.at<float>(i, j) = image.at<float>(i, j + 1) - image.at<float>(i, j);
        }
        // 边界条件：gradX(:, end) = image(:, 1) - image(:, end)
        gradX.at<float>(i, image.cols - 1) = image.at<float>(i, 0) - image.at<float>(i, image.cols - 1);
    }

    return gradX;
}

// y方向的一阶向前有限差分
cv::Mat forwardDiffY(const cv::Mat& image) {
    CV_Assert(image.type() == CV_32F || image.type() == CV_64F);

    cv::Mat gradY = cv::Mat::zeros(image.size(), image.type());

    // 计算差分：gradY(1:end-1, :) = image(2:end, :) - image(1:end-1, :)
    for (int i = 0; i < image.rows - 1; i++) {
        for (int j = 0; j < image.cols; j++) {
            gradY.at<float>(i, j) = image.at<float>(i + 1, j) - image.at<float>(i, j);
        }
    }
    // 边界条件：gradY(end, :) = image(1, :) - image(end, :)
    for (int j = 0; j < image.cols; j++) {
        gradY.at<float>(image.rows - 1, j) = image.at<float>(0, j) - image.at<float>(image.rows - 1, j);
    }

    return gradY;
}

// 计算散度
cv::Mat divergenceXY(const cv::Mat& X, const cv::Mat& Y) {
    CV_Assert(X.size() == Y.size() && (X.type() == CV_32F || X.type() == CV_64F));
    CV_Assert(Y.type() == X.type());

    cv::Mat divXY = cv::Mat::zeros(X.size(), X.type());

    // X方向散度：[X(:, end) - X(:, 1), -diff(X, 1, 2)]
    cv::Mat diffX;
    cv::Mat X_diff(X.rows, X.cols - 1, X.type());
    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < X.cols - 1; j++) {
            X_diff.at<float>(i, j) = X.at<float>(i, j + 1) - X.at<float>(i, j);
        }
    }

    cv::Mat X_boundary = cv::Mat::zeros(X.rows, X.cols, X.type());
    for (int i = 0; i < X.rows; i++) {
        X_boundary.at<float>(i, 0) = X.at<float>(i, X.cols - 1) - X.at<float>(i, 0);
        for (int j = 1; j < X.cols; j++) {
            X_boundary.at<float>(i, j) = -X_diff.at<float>(i, j - 1);
        }
    }

    // Y方向散度：[Y(end, :) - Y(1, :); -diff(Y, 1, 1)]
    cv::Mat diffY;
    cv::Mat Y_diff(Y.rows - 1, Y.cols, Y.type());
    for (int i = 0; i < Y.rows - 1; i++) {
        for (int j = 0; j < Y.cols; j++) {
            Y_diff.at<float>(i, j) = Y.at<float>(i + 1, j) - Y.at<float>(i, j);
        }
    }

    cv::Mat Y_boundary = cv::Mat::zeros(Y.rows, Y.cols, Y.type());
    for (int j = 0; j < Y.cols; j++) {
        Y_boundary.at<float>(0, j) = Y.at<float>(Y.rows - 1, j) - Y.at<float>(0, j);
        for (int i = 1; i < Y.rows; i++) {
            Y_boundary.at<float>(i, j) = -Y_diff.at<float>(i - 1, j);
        }
    }

    // 总散度
    divXY = X_boundary + Y_boundary;

    return divXY;
}

// 计算频域系数
FreqCoeff getFreqCoeff(const cv::Mat& inputImage) {
    FreqCoeff freqCoeff;

    // 获取图像尺寸
    cv::Size imageSize = inputImage.size();

    // 创建x方向差分核 [1, -1]
    cv::Mat kernelX = (cv::Mat_<float>(1, 2) << 1.0f, -1.0f);

    // 创建y方向差分核 [1; -1]
    cv::Mat kernelY = (cv::Mat_<float>(2, 1) << 1.0f, -1.0f);

    // 计算OTF (Optical Transfer Function)
    cv::Mat otfX, otfY;
    cv::dft(kernelX, otfX, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);
    cv::dft(kernelY, otfY, cv::DFT_COMPLEX_OUTPUT);

    // 调整到图像尺寸
    cv::Mat otfX_padded, otfY_padded;
    cv::copyMakeBorder(otfX, otfX_padded, 0, imageSize.height - otfX.rows, 0, imageSize.width - otfX.cols,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0));
    cv::copyMakeBorder(otfY, otfY_padded, 0, imageSize.height - otfY.rows, 0, imageSize.width - otfY.cols,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0));

    // 计算特征值（幅度平方）
    std::vector<cv::Mat> planesX(2), planesY(2);
    cv::split(otfX_padded, planesX);
    cv::split(otfY_padded, planesY);

    cv::Mat magX, magY;
    cv::magnitude(planesX[0], planesX[1], magX);
    cv::magnitude(planesY[0], planesY[1], magY);

    freqCoeff.eigsGradXtGradX = magX.mul(magX);
    freqCoeff.eigsGradYtGradY = magY.mul(magY);

    return freqCoeff;
}

// 图像块转换为矩阵
cv::Mat im2Patch(const cv::Mat& im, int win) {
    CV_Assert(im.channels() == 1 && (im.type() == CV_32F || im.type() == CV_64F));

    int N = im.rows - win + 1;
    int M = im.cols - win + 1;
    int L = N * M;

    cv::Mat X(win * win, L, im.type());

    int k = 0;
    for (int i = 0; i < win; i++) {
        for (int j = 0; j < win; j++) {
            k++;
            cv::Mat blk = im(cv::Range(i, i + N), cv::Range(j, j + M));

            // 手动转换为行向量，避免reshape问题
            cv::Mat row_vec(1, L, im.type());
            for (int r = 0; r < N; r++) {
                for (int c = 0; c < M; c++) {
                    int idx = r + c * N;  // 列优先顺序
                    if (idx < L) {
                        row_vec.at<float>(0, idx) = blk.at<float>(r, c);
                    }
                }
            }

            row_vec.copyTo(X.row(k - 1));
        }
    }

    return X;
}

// 块匹配算法
cv::Mat blockMatching(const cv::Mat& im, int win, int step, int nblk) {
    const int S = 21;
    int f = win;
    int f2 = f * f;
    int s = step;

    int N = im.rows - f + 1;
    int M = im.cols - f + 1;

    // 生成参考位置
    std::vector<int> r, c;
    for (int i = 1; i <= N; i += s) {
        r.push_back(i - 1);  // 转换为0-based索引
    }
    if (r.back() < N - 1) {
        r.push_back(N - 1);
    }

    for (int i = 1; i <= M; i += s) {
        c.push_back(i - 1);  // 转换为0-based索引
    }
    if (c.back() < M - 1) {
        c.push_back(M - 1);
    }

    int L = N * M;
    cv::Mat X = im2Patch(im, f);
    X = X.t();  // 转置为 L x f2

    cv::Mat I = cv::Mat::zeros(N, M, CV_32S);
    for (int i = 0; i < L; i++) {
        int row = i % N;
        int col = i / N;
        I.at<int>(row, col) = i;
    }

    // 确保矩阵连续
    if (!I.isContinuous()) {
        I = I.clone();
    }

    size_t N1 = r.size();
    size_t M1 = c.size();
    cv::Mat posArr(nblk, static_cast<int>(N1 * M1), CV_32S);

    for (size_t i = 0; i < N1; i++) {
        for (size_t j = 0; j < M1; j++) {
            int row = r[i];
            int col = c[j];
            int off = col * N + row;
            int off1 = j * N1 + i;

            int rmin = std::max(row - S, 0);
            int rmax = std::min(row + S, N - 1);
            int cmin = std::max(col - S, 0);
            int cmax = std::min(col + S, M - 1);

            cv::Mat idx = I(cv::Range(rmin, rmax + 1), cv::Range(cmin, cmax + 1));
            // 手动展平矩阵
            cv::Mat idx_flat(1, idx.rows * idx.cols, CV_32S);
            for (int ii = 0; ii < idx.rows; ii++) {
                for (int jj = 0; jj < idx.cols; jj++) {
                    idx_flat.at<int>(0, ii * idx.cols + jj) = idx.at<int>(ii, jj);
                }
            }
            idx = idx_flat;

            cv::Mat B(idx.rows, f2, im.type());
            for (int k = 0; k < idx.rows; k++) {
                int patch_idx = idx.at<int>(k);
                X.row(patch_idx).copyTo(B.row(k));
            }

            cv::Mat v = X.row(off);

            cv::Mat dis = cv::Mat::zeros(B.rows, 1, CV_32F);
            for (int k = 0; k < f2; k++) {
                cv::Mat diff = B.col(k) - v.at<float>(k);
                dis += diff.mul(diff);
            }

            std::vector<float> dis_vec;
            dis_vec.reserve(dis.rows);
            for (int i = 0; i < dis.rows; i++) {
                dis_vec.push_back(dis.at<float>(i, 0));
            }

            std::vector<size_t> indices(dis_vec.size());
            for (size_t k = 0; k < indices.size(); k++) indices[k] = k;
            std::sort(indices.begin(), indices.end(),
                     [&dis_vec](size_t a, size_t b) { return dis_vec[a] < dis_vec[b]; });

            for (int k = 0; k < std::min(nblk, (int)indices.size()); k++) {
                posArr.at<int>(k, off1) = idx.at<int>(indices[k]);
            }
        }
    }

    return posArr;
}

// 低秩SVD估计
void lowRankSvdEstimation(const cv::Mat& Y, double nsig, cv::Mat& X, cv::Mat& W, int& r) {
    // 检查输入矩阵是否有nan或inf
    cv::Mat Y_check = Y.clone();
    cv::patchNaNs(Y_check, 0.0);

    cv::Mat U, S, Vt;
    cv::SVDecomp(Y_check, S, U, Vt, cv::SVD::FULL_UV);

    cv::Mat S_diag;
    if (S.cols == 1) {
        S_diag = S.clone();
    } else {
        S_diag = S.diag();
    }

    cv::Mat S2 = S_diag.mul(S_diag);

    int ms = Y.rows;
    int ns = Y.cols;
    int num = S_diag.rows;

    double ss = 0.0;
    r = 1;

    for (int j = num - 1; j >= 0; j--) {
        ss += S2.at<float>(j);
        double sk = ss / (ms * ns);
        if (sk > nsig * nsig) {
            r = j + 2;
            break;
        }
    }

    if (r > num) r = num;

    cv::Mat U_r = U.colRange(0, r);
    cv::Mat Vt_r = Vt.rowRange(0, r);
    cv::Mat S_r = cv::Mat::diag(S_diag.rowRange(0, r));

    X = U_r * S_r * Vt_r;

    double wei;
    if (r == ms) {
        wei = 1.0 / std::max(1, ms);
    } else {
        wei = std::max(0.0, (ms - r) / (double)std::max(1, ms));
    }

    W = cv::Mat::ones(X.size(), X.type()) * wei;
    X = X * wei;

    // 确保结果不包含nan
    cv::patchNaNs(X, 0.0);
    cv::patchNaNs(W, wei);
}
