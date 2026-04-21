function [features, featureInfo] = extractFeatures(processedImg)
    % 特征维度常量
    FEATURE_DIMS = struct(...
        'geo', 8, ...          % 几何特征维度
        'hu', 7, ...           % Hu矩特征维度
        'fourier', 10, ...     % 傅里叶描述子
        'shape', 20);          % 形状上下文特征
    
    try
        % 1. 提取几何特征
        stats = regionprops(processedImg > 0, ...
            'Area', 'Perimeter', 'Centroid', ...
            'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', ...
            'Orientation', 'Solidity', 'ConvexArea', 'Extent');
        
        if isempty(stats)
            geoFeatures = zeros(1, FEATURE_DIMS.geo);
            huFeatures = zeros(1, FEATURE_DIMS.hu);
            fourierFeatures = zeros(1, FEATURE_DIMS.fourier);
            shapeFeatures = zeros(1, FEATURE_DIMS.shape);
        else
            % 选择最大连通区域
            if length(stats) > 1
                areas = [stats.Area];
                [~, maxIdx] = max(areas);
                stats = stats(maxIdx);
            end
            
            % 增强的几何特征
            imgSize = numel(processedImg);
            compactness = 4 * pi * stats.Area / (stats.Perimeter ^ 2);
            convexity = stats.Area / stats.ConvexArea;
            
            geoFeatures = [
                double(stats.Area) / imgSize, ...
                double(stats.Perimeter) / sqrt(imgSize), ...
                compactness, ...
                convexity, ...
                double(stats.Eccentricity), ...
                double(stats.Solidity), ...
                double(stats.Extent), ...
                double(stats.Orientation) / 180
            ];
            
            % 2. 提取Hu矩特征
            huFeatures = calculateHuMoments(processedImg);
            
            % 3. 提取傅里叶描述子
            fourierFeatures = extractFourierDescriptors(processedImg, FEATURE_DIMS.fourier);
            
            % 4. 提取形状上下文特征
            shapeFeatures = extractShapeContext(processedImg, FEATURE_DIMS.shape);
        end
        
        % 特征归一化
        geoFeatures = normalizeFeatures(geoFeatures);
        huFeatures = normalizeFeatures(huFeatures);
        fourierFeatures = normalizeFeatures(fourierFeatures);
        shapeFeatures = normalizeFeatures(shapeFeatures);
        
        % 组合所有特征
        features = [geoFeatures, huFeatures, fourierFeatures, shapeFeatures];
        
        % 返回特征信息
        if nargout > 1
            featureInfo = struct(...
                'geoDimension', FEATURE_DIMS.geo, ...
                'huDimension', FEATURE_DIMS.hu, ...
                'fourierDimension', FEATURE_DIMS.fourier, ...
                'shapeDimension', FEATURE_DIMS.shape, ...
                'totalDimension', sum(struct2array(FEATURE_DIMS)));
        end
        
    catch e
        errorMsg = sprintf('特征提取失败:\n位置: %s\n原因: %s', ...
            e.stack(1).name, e.message);
        error(errorMsg);
    end
end

function fourierFeatures = extractFourierDescriptors(img, numDescriptors)
    % 提取边界
    B = bwboundaries(img);
    if isempty(B)
        fourierFeatures = zeros(1, numDescriptors);
        return;
    end
    
    % 使用最长的边界
    [~, idx] = max(cellfun(@length, B));
    boundary = B{idx};
    
    % 计算傅里叶描述子
    complexBoundary = complex(boundary(:,2), boundary(:,1));
    fourierDesc = fft(complexBoundary);
    
    % 归一化
    fourierDesc = abs(fourierDesc(2:end)) / abs(fourierDesc(1));
    
    % 取前numDescriptors个描述子
    if length(fourierDesc) >= numDescriptors
        fourierFeatures = fourierDesc(1:numDescriptors)';
    else
        fourierFeatures = [fourierDesc; zeros(numDescriptors-length(fourierDesc), 1)]';
    end
end

function huMoments = calculateHuMoments(img)
    % 计算Hu不变矩
    img = double(img);
    [height, width] = size(img);
    [x, y] = meshgrid(1:width, 1:height);
    x = x(:); y = y(:); img = img(:);
    
    m00 = sum(img);
    if m00 > 0
        m10 = sum(x .* img);
        m01 = sum(y .* img);
        xc = m10/m00;
        yc = m01/m00;
        
        mu20 = sum((x - xc).^2 .* img) / m00;
        mu02 = sum((y - yc).^2 .* img) / m00;
        mu11 = sum((x - xc).*(y - yc) .* img) / m00;
        mu30 = sum((x - xc).^3 .* img) / m00;
        mu03 = sum((y - yc).^3 .* img) / m00;
        mu21 = sum((x - xc).^2 .*(y - yc) .* img) / m00;
        mu12 = sum((x - xc).*(y - yc).^2 .* img) / m00;
        
        huMoments = zeros(1,7);
        huMoments(1) = mu20 + mu02;
        huMoments(2) = (mu20 - mu02)^2 + 4*mu11^2;
        huMoments(3) = (mu30 - 3*mu12)^2 + (3*mu21 - mu03)^2;
        huMoments(4) = (mu30 + mu12)^2 + (mu21 + mu03)^2;
        huMoments(5) = (mu30 - 3*mu12)*(mu30 + mu12)*((mu30 + mu12)^2 - ...
            3*(mu21 + mu03)^2) + (3*mu21 - mu03)*(mu21 + mu03)*(3*(mu30 + mu12)^2 - ...
            (mu21 + mu03)^2);
        huMoments(6) = (mu20 - mu02)*((mu30 + mu12)^2 - (mu21 + mu03)^2) + ...
            4*mu11*(mu30 + mu12)*(mu21 + mu03);
        huMoments(7) = (3*mu21 - mu03)*(mu30 + mu12)*((mu30 + mu12)^2 - ...
            3*(mu21 + mu03)^2) - (mu30 - 3*mu12)*(mu21 + mu03)*(3*(mu30 + mu12)^2 - ...
            (mu21 + mu03)^2);
        
        huMoments = -sign(huMoments) .* log10(abs(huMoments) + eps);
    else
        huMoments = zeros(1,7);
    end
end

function normalizedFeatures = normalizeFeatures(features)
    features(isnan(features)) = 0;
    mu = mean(features);
    sigma = std(features);
    
    if sigma > 0
        normalizedFeatures = (features - mu) / sigma;
    else
        normalizedFeatures = features - mu;
    end
    
    normalizedFeatures(isnan(normalizedFeatures)) = 0;
    normalizedFeatures(isinf(normalizedFeatures)) = 0;
    normalizedFeatures = max(min(normalizedFeatures, 5), -5);
end

function shapeFeatures = extractShapeContext(img, numFeatures)
    % 提取边界点
    [B,~] = bwboundaries(img, 'noholes');
    if isempty(B)
        shapeFeatures = zeros(1, numFeatures);
        return;
    end
    
    % 使用最长的边界
    [~, idx] = max(cellfun(@length, B));
    boundary = B{idx};
    
    % 采样边界点
    numPoints = min(50, size(boundary, 1));
    sampledPoints = boundary(round(linspace(1, size(boundary,1), numPoints)), :);
    
    % 计算形状描述子
    bins_r = 4;      % 径向分区数
    bins_theta = 5;  % 角度分区数
    
    % 初始化形状描述子
    descriptor = zeros(1, bins_r * bins_theta);
    
    % 计算中心点
    center = mean(sampledPoints, 1);
    
    % 计算最大距离（用于归一化）
    maxDist = max(sqrt(sum((sampledPoints - center).^2, 2)));
    
    % 对每个采样点计算相对分布
    for i = 1:numPoints
        % 计算相对于中心点的距离和角度
        dx = sampledPoints(i,2) - center(2);
        dy = sampledPoints(i,1) - center(1);
        r = sqrt(dx^2 + dy^2) / maxDist;
        theta = atan2(dy, dx);
        
        % 确定bin索引
        r_idx = min(floor(r * bins_r) + 1, bins_r);
        theta_idx = mod(floor((theta + pi)/(2*pi) * bins_theta), bins_theta) + 1;
        
        % 更新直方图
        bin_idx = (r_idx-1)*bins_theta + theta_idx;
        descriptor(bin_idx) = descriptor(bin_idx) + 1;
    end
    
    % 归一化描述子
    descriptor = descriptor / sum(descriptor);
    
    % 调整维度
    if length(descriptor) >= numFeatures
        shapeFeatures = descriptor(1:numFeatures);
    else
        shapeFeatures = [descriptor, zeros(1, numFeatures-length(descriptor))];
    end
end 