function restored_img = reverse_filter(original_path, filtered_path)
    filterList = {
        fspecial('gaussian', [7 7], 2), ...
        fspecial('log', [5 5], 0.5), ...
        fspecial('motion', 10, 45), ...
        fspecial('disk', 5), ...
        % fspecial('unsharp', 0.5)
    };
    filterNames = {'Gaussian', 'LoG', 'Motion', 'Disk'}; %, 'Unsharp'
    
    % Load images
    cleanRGB = im2double(imread(original_path));
    blurredRGB = im2double(imread(filtered_path));

    % Step 3: 尺寸适配和数据类型统一 Size adaptation and data type unification
    if ~isequal(size(cleanRGB), size(blurredRGB))
        warning('Images size not same, automatically adjust the filtered image to the original image size');
        blurredRGB = imresize(blurredRGB, [size(cleanRGB,1), size(cleanRGB,2)]);
    end

    % 确保都是双精度浮点数 Make sure all double precision floating point numbers
    cleanRGB = im2double(cleanRGB);
    blurredRGB = im2double(blurredRGB);

    % Step 4: 判断灰度或彩色 Determine grayscale or color
if size(cleanRGB,3) == 1  % 灰度图
    isColor = false;
    cleanGray = cleanRGB;
    blurGray = blurredRGB;
else % 彩色图
    isColor = true;
    % 提前转换为灰度用于SSIM计算
    cleanGray = rgb2gray(cleanRGB);
    blurGray = rgb2gray(blurredRGB);
end

bestSSIM = -inf;
bestRestored = [];
bestIndex = 0;
allRestored = cell(1, length(filterList));
allSSIM = zeros(1, length(filterList));
fprintf('Start reverse image filter...\n');

for k = 1:length(filterList)
    fprintf('Testing filter applied... %d/%d: %s\n', k, length(filterList), filterNames{k});
    
    H = filterList{k};
    
    % 添加正则化参数防止除零和数值不稳定 Add regularization parameters to prevent division by zero and numerical instability
    lambda = 1e-6;
    N = 15; % 适度减少迭代次数，避免过拟合 overfitting
    
    try
        if isColor
            % 分别处理RGB三个通道
            cleanR = cleanRGB(:,:,1); cleanG = cleanRGB(:,:,2); cleanB = cleanRGB(:,:,3);
            blurR = blurredRGB(:,:,1); blurG = blurredRGB(:,:,2); blurB = blurredRGB(:,:,3);
            
            Xcur_R = blurR; Xcur_G = blurG; Xcur_B = blurB;
            
            for i = 1:N
                % 使用边界处理避免循环卷积artifacts, Use edge processing to avoid circular convolution artifacts
                Xfcur_R = imfilter(Xcur_R, H, 'replicate');
                Xfcur_G = imfilter(Xcur_G, H, 'replicate');
                Xfcur_B = imfilter(Xcur_B, H, 'replicate');
                
                % 频域去卷积带正则化
                H_fft = psf2otf(H, size(Xcur_R));
                H_conj = conj(H_fft);
                H_abs2 = abs(H_fft).^2;
                
                Xcur_R = real(ifft2((fft2(blurR) .* H_conj) ./ (H_abs2 + lambda)));
                Xcur_G = real(ifft2((fft2(blurG) .* H_conj) ./ (H_abs2 + lambda)));
                Xcur_B = real(ifft2((fft2(blurB) .* H_conj) ./ (H_abs2 + lambda)));
                
                % 限制数值范围，避免异常值
                Xcur_R = max(0, min(1, Xcur_R));
                Xcur_G = max(0, min(1, Xcur_G));
                Xcur_B = max(0, min(1, Xcur_B));
            end
            
            deRGB = cat(3, Xcur_R, Xcur_G, Xcur_B);
            
        else
            % 灰度图处理
            Xcur = blurGray;
            
            for i = 1:N
                Xfcur = imfilter(Xcur, H, 'replicate');
                
                H_fft = psf2otf(H, size(Xcur));
                H_conj = conj(H_fft);
                H_abs2 = abs(H_fft).^2;
                
                Xcur = real(ifft2((fft2(blurGray) .* H_conj) ./ (H_abs2 + lambda)));
                Xcur = max(0, min(1, Xcur)); % 限制数值范围
            end
            
            deRGB = Xcur;
        end
        
        % 确保尺寸完全一致
        if ~isequal(size(deRGB), size(cleanRGB))
            deRGB = imresize(deRGB, [size(cleanRGB,1), size(cleanRGB,2)]);
        end
        
        allRestored{k} = deRGB;
        
        % SSIM评估 - 统一使用灰度图计算
        if isColor
            grayRestored = rgb2gray(deRGB);
        else
            grayRestored = deRGB;
        end
        
        % 确保SSIM计算的两张图片尺寸和数据类型一致
        if ~isequal(size(grayRestored), size(cleanGray))
            grayRestored = imresize(grayRestored, size(cleanGray));
        end
        
        % 计算SSIM，添加异常处理
        ssimVal = ssim(im2double(grayRestored), im2double(cleanGray));
        
        % 检查SSIM值的有效性
        if isnan(ssimVal) || isinf(ssimVal)
            warning('滤波器 %s 产生了无效的SSIM值', filterNames{k});
            ssimVal = -1; % 给一个很低的分数
        end
        
        allSSIM(k) = ssimVal;
        fprintf('Filter: %-8s  -> SSIM = %.4f\n', filterNames{k}, ssimVal);
        
        if ssimVal > bestSSIM
            bestSSIM = ssimVal;
            bestRestored = deRGB;
            bestIndex = k;
        end
        
    catch ME
        warning('滤波器 %s 处理失败: %s', filterNames{k}, ME.message);
        allSSIM(k) = -1; % 失败情况给最低分
        allRestored{k} = blurredRGB; % 用原模糊图作为占位符
    end

    if bestIndex > 0
        restored_img = bestRestored;
    else
        restored_img = blurredRGB;
    end
    
end
