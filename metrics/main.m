function main() 
    % 交互式选择红外图像和过滤后图像的文件夹
    infraredFolder = uigetdir([], '选择红外图像文件夹');
    filteredFolder = uigetdir([], '选择过滤后图像文件夹');
    
    % 获取两个文件夹中所有图像文件的列表
    infraredFiles = dir(fullfile(infraredFolder, '*.jpg'));
    filteredFiles = dir(fullfile(filteredFolder, '*.jpg'));

    % 检查文件数量是否一致
    if length(infraredFiles) ~= length(filteredFiles)
        error('红外图像和过滤后图像文件夹中的文件数量不一致');
    end
    
    % 创建日志文件
    logFile = 'log.txt';
    fid = fopen(logFile, 'a');
    
    % 遍历文件夹中的文件
    for i = 1:length(infraredFiles)
        % 读取红外图像和过滤后图像
        infraredImage = imread(fullfile(infraredFiles(i).folder, infraredFiles(i).name));
        filteredImage = imread(fullfile(filteredFiles(i).folder, filteredFiles(i).name));

        % 检查是否为彩色图像，并转换为灰度图像
        if size(infraredImage, 3) == 3  
            infraredImage = im2gray(infraredImage);  
        end  
        if size(filteredImage, 3) == 3  
            filteredImage = im2gray(filteredImage);  
        end  

        % 转换为双精度
        infraredImage = double(infraredImage);
        filteredImage = double(filteredImage);
        
        % 获取时间戳
        timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        
        % 计算各个指标
        ICV = calculateICV(filteredImage);
        MRD = calculateMRD(infraredImage, filteredImage);
        mseValue = calculateMSE(infraredImage, filteredImage);
        psnrValue = calculatePSNR(mseValue);
        ssimValue = calculateSSIM(infraredImage, filteredImage);
        AVGE = calculateAVGE(infraredImage, filteredImage);
        [infraredRoughness, filteredRoughness] = calculateRoughness(infraredImage, filteredImage);

        % 打印结果到命令行
        fprintf('处理图像: %s 和 %s\n', infraredFiles(i).name, filteredFiles(i).name);
        fprintf('ICV: %.4f, MRD: %.4f, MSE: %.4f, PSNR: %.4f dB, SSIM: %.4f, AVGE: %.4f\n', ...
                ICV, MRD, mseValue, psnrValue, ssimValue, AVGE);
        fprintf('原始图像粗糙度指数 ρ 值: %.4f, 过滤后图像粗糙度指数 ρ 值: %.4f\n\n', ...
                infraredRoughness, filteredRoughness);
        
        % 输出到日志文件
        fprintf(fid, '[%s] 处理图像: %s 和 %s\n', timestamp, infraredFiles(i).name, filteredFiles(i).name);
        fprintf(fid, 'ICV: %.4f, MRD: %.4f, MSE: %.4f, PSNR: %.4f dB, SSIM: %.4f, AVGE: %.4f\n', ...
                ICV, MRD, mseValue, psnrValue, ssimValue, AVGE);
        fprintf(fid, '原始图像粗糙度指数 ρ 值: %.4f, 过滤后图像粗糙度指数 ρ 值: %.4f\n\n', ...
                infraredRoughness, filteredRoughness);
    end
    fclose(fid);
end
