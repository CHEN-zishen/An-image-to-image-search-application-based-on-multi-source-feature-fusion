function main()
    % 添加所有子文件夹到路径
    addpath(genpath(pwd));
    
    % 清理工作空间
    clear;
    clc;
    
    % 检查数据集文件夹
    if ~exist('data', 'dir')
        error('未找到数据集文件夹，请确保"data"文件夹存在且包含图像文件。');
    end
    
    % 启动GUI界面
    imageRetrievalGUI();
end 