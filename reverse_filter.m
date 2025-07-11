function restored_img = reverse_filter(original_path, filtered_path)
    filterList = {
        fspecial('gaussian', [7 7], 2), ...
        fspecial('log', [5 5], 0.5), ...
        fspecial('motion', 10, 45), ...
        fspecial('disk', 5), ...
        [-0.2 -1 -0.2; -1 5.8 -1; -0.2 -1 -0.2]  % Unsharp filter
    };
    filterNames = {'Gaussian', 'LoG', 'Motion', 'Disk', 'Unsharp'};
    
    % Load images
    cleanRGB = im2double(imread(original_path));
    blurredRGB = im2double(imread(filtered_path));

    % Size adaptation and data type unification
    if ~isequal(size(cleanRGB), size(blurredRGB))
        warning('Images size not same, automatically adjust the filtered image to the original image size');
        blurredRGB = imresize(blurredRGB, [size(cleanRGB,1), size(cleanRGB,2)]);
    end

    % Make sure all double precision floating point numbers
    cleanRGB = im2double(cleanRGB);
    blurredRGB = im2double(blurredRGB);

    fprintf('Comparing original and filtered images...\n');
    psnr_val = psnr(cleanRGB, blurredRGB);
    fprintf('  PSNR: %.4f dB\n', psnr_val);
    psnr_threshold = 95;

    if (psnr_val > psnr_threshold)
        fprintf('Images are already identical or very similar - no filtering detected!\n');
        fprintf('Returning original filtered image without processing.\n');
        restored_img = blurredRGB;
        return;
    end
    
    fprintf('Filter detected - proceeding with reverse filtering algorithm...\n');

    % Determine grayscale or color
    if size(cleanRGB,3) == 1  
        isColor = false;
        cleanGray = cleanRGB;
        blurGray = blurredRGB;
    else 
        isColor = true;
        cleanGray = rgb2gray(cleanRGB);
        blurGray = rgb2gray(blurredRGB);
    end

    % PRE-FILTERING: Discriminate between blur vs sharpen filters
    fprintf('Analyzing filter characteristics...\n');
    [blur_score, sharp_score] = analyze_filter_type(cleanGray, blurGray);
    fprintf('Blur score: %.4f, Sharp score: %.4f\n', blur_score, sharp_score);
    
    % Select appropriate filter subset based on analysis
    if blur_score > sharp_score
        fprintf('Image appears to be BLURRED - excluding sharpening filters\n');
        active_filters = [1, 2, 3, 4]; % Gaussian, LoG, Motion, Disk
    else
        fprintf('Image appears to be SHARPENED - excluding blurring filters\n');
        active_filters = [5]; % Only Unsharp
        % You might want to add more sharpening filters here
    end
    
    % If the discrimination is not clear, test a smaller subset first
    if abs(blur_score - sharp_score) < 0.1
        fprintf('Filter type unclear - testing both categories\n');
        active_filters = 1:length(filterList);
    end

    bestSSIM = -inf;
    bestRestored = [];
    bestIndex = 0;
    allRestored = cell(1, length(filterList));
    allSSIM = zeros(1, length(filterList));
    fprintf('Start reverse image filter...\n');

    for k = active_filters
        fprintf('Testing filter applied... %d/%d: %s\n', k, length(filterList), filterNames{k});
        
        H = filterList{k};
        
        % Add regularization parameters to prevent division by zero and numerical instability
        lambda = 1e-6;
        N = 15; 
        
        try
            if isColor
                cleanR = cleanRGB(:,:,1); cleanG = cleanRGB(:,:,2); cleanB = cleanRGB(:,:,3);
                blurR = blurredRGB(:,:,1); blurG = blurredRGB(:,:,2); blurB = blurredRGB(:,:,3);
                
                Xcur_R = blurR; Xcur_G = blurG; Xcur_B = blurB;
                
                for i = 1:N
                    % Use edge processing to avoid circular convolution artifacts
                    Xfcur_R = imfilter(Xcur_R, H, 'replicate');
                    Xfcur_G = imfilter(Xcur_G, H, 'replicate');
                    Xfcur_B = imfilter(Xcur_B, H, 'replicate');
                    
                    % Frequency Domain Deconvolution with Regularization
                    H_fft = psf2otf(H, size(Xcur_R));
                    H_conj = conj(H_fft);
                    H_abs2 = abs(H_fft).^2;
                    
                    Xcur_R = real(ifft2((fft2(blurR) .* H_conj) ./ (H_abs2 + lambda)));
                    Xcur_G = real(ifft2((fft2(blurG) .* H_conj) ./ (H_abs2 + lambda)));
                    Xcur_B = real(ifft2((fft2(blurB) .* H_conj) ./ (H_abs2 + lambda)));
                    
                    Xcur_R = max(0, min(1, Xcur_R));
                    Xcur_G = max(0, min(1, Xcur_G));
                    Xcur_B = max(0, min(1, Xcur_B));
                end
                
                deRGB = cat(3, Xcur_R, Xcur_G, Xcur_B);
                deRGB = medfilt3(deRGB, [3 3 1]);
                
            else
                Xcur = blurGray;
                
                for i = 1:N
                    Xfcur = imfilter(Xcur, H, 'replicate');
                    
                    H_fft = psf2otf(H, size(Xcur));
                    H_conj = conj(H_fft);
                    H_abs2 = abs(H_fft).^2;
                    
                    Xcur = real(ifft2((fft2(blurGray) .* H_conj) ./ (H_abs2 + lambda)));
                    Xcur = max(0, min(1, Xcur)); 
                end
                
                deRGB = Xcur;
                deRGB = medfilt2(deRGB, [3 3]);
            end
            
            if ~isequal(size(deRGB), size(cleanRGB))
                deRGB = imresize(deRGB, [size(cleanRGB,1), size(cleanRGB,2)]);
            end
            
            allRestored{k} = deRGB;
            
            if isColor
                grayRestored = rgb2gray(deRGB);
            else
                grayRestored = deRGB;
            end
            
            if ~isequal(size(grayRestored), size(cleanGray))
                grayRestored = imresize(grayRestored, size(cleanGray));
            end
            
            ssimVal = ssim(im2double(grayRestored), im2double(cleanGray));
            
            if isnan(ssimVal) || isinf(ssimVal)
                warning('Invalid SSIM value for %s', filterNames{k});
                ssimVal = -1; 
            end
            
            allSSIM(k) = ssimVal;
            fprintf('Filter: %-8s  -> SSIM = %.4f\n', filterNames{k}, ssimVal);
            
            if ssimVal > bestSSIM
                bestSSIM = ssimVal;
                bestRestored = deRGB;
                bestIndex = k;
            end
            
        catch ME
            warning('Filter processing failed for %s: %s', filterNames{k}, ME.message);
            allSSIM(k) = -1;
            allRestored{k} = blurredRGB; 
        end
    end

    % Final result assignment
    if bestIndex > 0
        restored_img = bestRestored;
        fprintf('Best filter identified: %s (SSIM = %.4f)\n', filterNames{bestIndex}, bestSSIM);
    else
        restored_img = blurredRGB;
        fprintf('No suitable filter found, returning original filtered image.\n');
    end
end

% Helper function to analyze filter type
function [blur_score, sharp_score] = analyze_filter_type(original, filtered)
    % Calculate gradients to measure sharpness
    [Gx_orig, Gy_orig] = gradient(original);
    [Gx_filt, Gy_filt] = gradient(filtered);
    
    % Calculate gradient magnitudes
    grad_mag_orig = sqrt(Gx_orig.^2 + Gy_orig.^2);
    grad_mag_filt = sqrt(Gx_filt.^2 + Gy_filt.^2);
    
    % Calculate mean gradient magnitudes
    mean_grad_orig = mean(grad_mag_orig(:));
    mean_grad_filt = mean(grad_mag_filt(:));
    
    % Calculate high-frequency content using FFT
    fft_orig = fft2(original);
    fft_filt = fft2(filtered);
    
    % Create high-frequency mask (outer region of frequency domain)
    [M, N] = size(original);
    [X, Y] = meshgrid(1:N, 1:M);
    center_x = N/2; center_y = M/2;
    radius = min(M, N) / 4;
    high_freq_mask = sqrt((X - center_x).^2 + (Y - center_y).^2) > radius;
    
    % Calculate high-frequency energy
    hf_energy_orig = sum(abs(fft_orig(high_freq_mask)).^2);
    hf_energy_filt = sum(abs(fft_filt(high_freq_mask)).^2);
    
    % Calculate scores
    % Blur score: higher if filtered image has less edge content and HF energy
    blur_score = (mean_grad_orig - mean_grad_filt) / mean_grad_orig + ...
                 (hf_energy_orig - hf_energy_filt) / hf_energy_orig;
    
    % Sharp score: higher if filtered image has more edge content and HF energy  
    sharp_score = (mean_grad_filt - mean_grad_orig) / mean_grad_orig + ...
                  (hf_energy_filt - hf_energy_orig) / hf_energy_orig;
    
    % Normalize scores to [0,1] range
    blur_score = max(0, min(1, blur_score));
    sharp_score = max(0, min(1, sharp_score));
end