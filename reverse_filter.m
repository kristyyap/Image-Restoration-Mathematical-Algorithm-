function [restored_img, filtered_img] = reverse_filter(inpath, filter_type)
    RGB = imread(inpath);
    R = im2double(RGB(:,:,1));
    G = im2double(RGB(:,:,2));
    B = im2double(RGB(:,:,3));

    switch lower(filter_type)
        case 'motion'
            H = fspecial('motion',10,45);
            f = @(x) imfilter(x,H,'circular');
        case 'gaussian'
            H = fspecial('gaussian', [7 7], 2);
            f = @(x) imfilter(x,H,'circular');
        case 'log'
            H = fspecial('log', [5 5], 0.5);
            f = @(x) imfilter(x,H,'circular');
        case 'disk'
            H = fspecial('disk',5);
            f = @(x) imfilter(x,H,'circular');
        case 'median'
            f = @(x) medfilt2(x);
        case 'wiener'
            f = @(x) wiener2(x, [5 5], (10/255)^2);
        otherwise
            H = fspecial('motion',10,45);
            f = @(x) imfilter(x,H,'circular');
    end

    Ry = f(R); Gy = f(G); By = f(B);
    filtered_img = cat(3, Ry, Gy, By);

    N = 20;
    Xcur_R = Ry; Xcur_G = Gy; Xcur_B = By;
    for i = 1:N
        Xfcur_R = f(Xcur_R);
        Xfcur_G = f(Xcur_G);
        Xfcur_B = f(Xcur_B);

        Xcur_R = ifft2((fft2(Ry).*fft2(Xcur_R))./(fft2(Xfcur_R)+eps));
        Xcur_G = ifft2((fft2(Gy).*fft2(Xcur_G))./(fft2(Xfcur_G)+eps));
        Xcur_B = ifft2((fft2(By).*fft2(Xcur_B))./(fft2(Xfcur_B)+eps));
    end
    restored_img = cat(3, Xcur_R, Xcur_G, Xcur_B);
end
