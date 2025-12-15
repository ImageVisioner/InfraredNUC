
## 算法原理

### 1. ADMM去条纹算法

采用ADMM优化框架将原始图像分解为清洁图像和条纹噪声分量：

```
I = U + S
```

其中：
- `I`: 原始噪声图像
- `U`: 清洁图像分量
- `S`: 条纹噪声分量



### 2. 降噪算法

对去条纹后的图像进行进一步降噪，使用基于块匹配和低秩近似的SVD方法：

- 块匹配：为每个图像块找到最相似的K个块
- 低秩近似：对相似块组进行SVD分解，保留主要奇异值
- 加权聚合：将降噪后的块加权聚合回完整图像

## 文件说明

### 主要文件

- `test1.m`: 主程序入口，实现完整的图像处理流程
- `Functions/`: 函数文件夹
  - `admmAlgorithm.m`: ADMM去条纹算法
  - `lraSvdDenoising.m`: LRA-SVD降噪算法
  - `getFreqCoeff.m`: 频域系数计算
  - `forwardDiffX.m`: x方向差分
  - `forwardDiffY.m`: y方向差分
  - `divergenceXY.m`: 散度计算
  - 其他辅助函数...
- `8.jpg`: 测试图像
- `README.md`: 项目说明文档

### 函数说明

#### Functions 文件夹中的函数

- `admmAlgorithm()`: ADMM去条纹主算法
- `lraSvdDenoising()`: LRA-SVD降噪算法
- `getFreqCoeff()`: 计算频域卷积系数
- `forwardDiffX()`: x方向一阶向前差分
- `forwardDiffY()`: y方向一阶向前差分
- `divergenceXY()`: 散度计算
- `im2Patch()`: 图像到块矩阵转换
- `blockMatching()`: 块匹配算法
- `lowRankSvdEstimation()`: 低秩SVD估计

## 使用方法

### 环境要求

- MATLAB R2018b 或更高版本
- Image Processing Toolbox

### 运行步骤

1. 确保测试图像 `8.jpg` 在工作目录中
2. 运行主脚本：
   ```matlab
   test1
   ```

### 参数设置

#### ADMM参数

```matlab
opts.lamda1 = 5*10^(-3);    % 列方向条纹正则化参数
opts.lamda2 = 1*10^(-3);    % 行方向稀疏正则化参数
opts.lamda3 = 2*10^(-3);    % 梯度一致性正则化参数
opts.beta1 = 0.05;          % 行方向增广拉格朗日参数
opts.beta2 = 0.1;           % 列方向增广拉格朗日参数
opts.beta3 = 0.08;          % 梯度方向增广拉格朗日参数
opts.tol = 1.e-4;           % 收敛阈值
opts.maxitr = 100;          % 最大迭代次数
```

#### LRA-SVD参数

```matlab
sigma = 1.0;  % 噪声标准差
```

参数会根据噪声水平自动调整。

## 输出结果

### 可视化结果

- 原始条纹图像
- 去条纹后图像
- 估计的条纹噪声
- 最终降噪结果
- 残留噪声分布

### 保存文件

- `destriped_image.png`: 去条纹结果
- `final_denoised_image.png`: 最终降噪结果


