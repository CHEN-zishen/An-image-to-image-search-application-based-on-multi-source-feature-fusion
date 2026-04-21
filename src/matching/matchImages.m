function [sortedIndices, similarities] = matchImages(queryFeatures, datasetFeatures)
    % 特征维度参数
    GEOM_DIM = 8;
    HU_DIM = 7;
    FOURIER_DIM = 10;
    SHAPE_DIM = 20;  % 新增形状特征维度
    TOTAL_DIM = GEOM_DIM + HU_DIM + FOURIER_DIM + SHAPE_DIM;
    
    try
        % 检查特征维度
        if size(queryFeatures,2) ~= TOTAL_DIM || size(datasetFeatures,2) ~= TOTAL_DIM
            error('特征维度不匹配：期望%d维，实际为查询%d维，数据集%d维', ...
                TOTAL_DIM, size(queryFeatures,2), size(datasetFeatures,2));
        end
        
        % 分离特征
        query_geo = queryFeatures(1:GEOM_DIM);
        query_hu = queryFeatures(GEOM_DIM+1:GEOM_DIM+HU_DIM);
        query_fourier = queryFeatures(GEOM_DIM+HU_DIM+1:end);
        
        similarities = zeros(size(datasetFeatures, 1), 1);
        
        for i = 1:size(datasetFeatures, 1)
            % 分离数据集特征
            dataset_geo = datasetFeatures(i, 1:GEOM_DIM);
            dataset_hu = datasetFeatures(i, GEOM_DIM+1:GEOM_DIM+HU_DIM);
            dataset_fourier = datasetFeatures(i, GEOM_DIM+HU_DIM+1:end);
            
            % 1. 几何特征相似度
            geo_diff = query_geo - dataset_geo;
            geo_sim = exp(-norm(geo_diff .* [0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02]));
            
            % 2. Hu矩相似度
            hu_diffs = abs(query_hu - dataset_hu);
            hu_sim = mean(exp(-3 * hu_diffs));
            
            % 3. 傅里叶描述子相似度
            fourier_sim = exp(-norm(query_fourier - dataset_fourier) / FOURIER_DIM);
            
            % 4. 检查是否为完全相同的图像
            if norm(geo_diff) < 1e-10 && norm(hu_diffs) < 1e-10 && ...
               norm(query_fourier - dataset_fourier) < 1e-10
                similarities(i) = 100;  % 完全匹配
                continue;
            end
            
            % 5. 自适应权重组合（对非完全匹配的情况）
            geo_weight = 0.5 + 0.1 * geo_sim;
            hu_weight = 0.3 + 0.1 * hu_sim;
            fourier_weight = 1 - (geo_weight + hu_weight);
            
            raw_similarity = geo_weight * geo_sim + ...
                           hu_weight * hu_sim + ...
                           fourier_weight * fourier_sim;
            
            % 6. 使用改进的sigmoid函数
            similarities(i) = 100 / (1 + exp(-12 * (raw_similarity - 0.4)));
        end
        
        % 排序前的预处理
        % 1. 移除异常值（保护完全匹配）
        perfect_matches = similarities == 100;
        mean_sim = mean(similarities(similarities > 0 & similarities < 100));
        std_sim = std(similarities(similarities > 0 & similarities < 100));
        similarities(~perfect_matches & similarities > mean_sim + 3*std_sim) = ...
            mean_sim + 3*std_sim;
        
        % 2. 排序
        [similarities, sortedIndices] = sort(similarities, 'descend');
        
        % 3. 相对化处理（改进版，保护完全匹配）
        if length(similarities) > 1 && similarities(1) < 100
            max_sim = similarities(1);
            
            % 根据最高相似度调整策略
            if max_sim > 85  % 非常相似
                % 温和的衰减
                decay = exp(-(0:length(similarities)-1)/3);
                similarities = similarities .* decay';
                
            elseif max_sim > 70  % 比较相似
                % 轻微衰减
                ratio = similarities(2:end) / max_sim;
                similarities(2:end) = similarities(2:end) .* (1 - 0.1 * ratio);
            end
        end
        
        % 4. 动态阈值处理
        threshold = 15;
        if max(similarities) < 50 && max(similarities) >= 30
            threshold = 10;
        end
        similarities(similarities < threshold) = 0;
        
        % 5. 相似度分布优化（仅对非完全匹配）
        non_perfect = similarities < 100;
        if any(non_perfect) && max(similarities(non_perfect)) > 0
            sim_adjust = similarities(non_perfect);
            sim_adjust = sim_adjust .^ (1 + (95 - max(sim_adjust))/200);
            
            if max(sim_adjust) > 0
                sim_adjust = sim_adjust * (95/max(sim_adjust));
            end
            similarities(non_perfect) = sim_adjust;
        end
        
        % 6. 结果分布优化（跳过完全匹配的情况）
        if length(similarities) > 1 && similarities(1) < 100
            diffs = diff(similarities(1:min(5,end)));
            mean_diff = mean(diffs);
            
            if mean_diff < 3 && similarities(1) > 60
                for i = 2:min(5,length(similarities))
                    similarities(i) = similarities(i) * (0.97 - 0.005*(i-1));
                end
            end
        end
        
        % 7. 最终限制
        similarities = min(max(similarities, 0), 100);  % 允许100%的相似度
        
    catch e
        fprintf('匹配错误：%s\n', e.message);
        rethrow(e);
    end
end 