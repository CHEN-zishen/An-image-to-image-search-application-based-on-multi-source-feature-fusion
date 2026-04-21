function processedImg = preprocessImage(inputImg)
    % 预处理参数
    params = struct(...
        'targetSize', [224 224], ...
        'morphClose', true, ...
        'boundingBox', true, ...
        'keepAspectRatio', true, ...
        'binarize', true, ...
        'edgePadding', true, ...
        'holeFilling', true, ...
        'noiseRemoval', true);    % 新增噪声去除
    
    try
        % 1. 转换为灰度图并增强对比度
        if size(inputImg, 3) == 3
            img = rgb2gray(inputImg);
        else
            img = inputImg;
        end
        
        % 确保图像是 uint8 类型
        if ~isa(img, 'uint8')
            img = uint8(img * 255);
        end
        
        % 2. 图像增强
        img = imadjust(img);  % 对比度增强
        img = wiener2(img, [3 3]);  % 维纳滤波去噪
        
        % 3. 自适应二值化
        if params.binarize
            T = graythresh(img);
            img = imbinarize(img, T);
        end
        
        % 4. 形态学处理
        if params.morphClose
            se1 = strel('disk', 3);
            se2 = strel('disk', 1);
            img = imclose(img, se1);
            img = imopen(img, se2);
        end
        
        % 5. 噪声去除
        if params.noiseRemoval
            img = bwareaopen(img, 50);  % 移除小面积对象
        end
        
        % 6. 包围盒处理
        if params.boundingBox
            stats = regionprops(img, 'BoundingBox', 'Area');
            if ~isempty(stats)
                [~, maxIdx] = max([stats.Area]);
                bbox = stats(maxIdx).BoundingBox;
                bbox = round(bbox);
                img = imcrop(img, bbox);
            end
        end
        
        % 7. 保持比例缩放
        if params.keepAspectRatio
            scale = min(params.targetSize ./ size(img));
            newSize = round(size(img) * scale);
            img = imresize(img, newSize);
            padSize = params.targetSize - newSize;
            padSize = floor(padSize/2);
            img = padarray(img, padSize, 0, 'both');
            
            % 处理奇数尺寸
            if size(img,1) ~= params.targetSize(1)
                img = padarray(img, [1 0], 0, 'post');
            end
            if size(img,2) ~= params.targetSize(2)
                img = padarray(img, [0 1], 0, 'post');
            end
        end
        
        % 8. 孔洞填充
        if params.holeFilling
            img = imfill(img, 'holes');
        end
        
        % 确保输出是逻辑类型
        processedImg = logical(img);
        
    catch e
        errorMsg = sprintf('图像预处理失败:\n位置: %s\n原因: %s', ...
            e.stack(1).name, e.message);
        error(errorMsg);
    end
end 