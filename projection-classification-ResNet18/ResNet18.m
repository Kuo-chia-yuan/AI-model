% 設定資料夾路徑
rootFolder = 'D:\Jalen\碩二上\感測器助教\BMP\ImageData_Rectangle\';

% 設定大類別 A, B, C, D, E
categories = {'A', 'B', 'C', 'D', 'E'};

% 設定一個空的圖像資料庫
allFiles = [];
allLabels = [];

% 遍歷每個大類別資料夾
for i = 1:length(categories)
    categoryFolder = fullfile(rootFolder, categories{i});
    
    % 尋找每個大類別資料夾下的所有子資料夾（Resion0 ~ Resion8）
    subfolders = dir(categoryFolder);
    subfolders = subfolders([subfolders.isdir]); % 只選擇資料夾
    
    % 遍歷每個子資料夾（Resion0 ~ Resion8）
    for j = 1:length(subfolders)
        if ~ismember(subfolders(j).name, {'.', '..'}) % 排除 "." 和 ".."
            resionFolder = fullfile(categoryFolder, subfolders(j).name);
            
            % 使用 imageDatastore 載入每個 Resion 資料夾中的圖片
            imds = imageDatastore(resionFolder, ...
                'LabelSource', 'foldernames');  % 標註來源為資料夾名稱
            
            % 將圖像和標籤合併到總資料集
            allFiles = [allFiles; imds.Files];
            allLabels = [allLabels; repmat(categories{i}, length(imds.Files), 1)];
        end
    end
end

% 將 allLabels 轉換為 categorical 類型
allLabels = categorical(cellstr(allLabels));

% 創建一個新的 imageDatastore，將所有圖片和標籤一起儲存
imds = imageDatastore(allFiles, 'Labels', allLabels);

% 加載預訓練的 ResNet-18 模型
net = resnet18;

% 將 ResNet-18 轉換為 layerGraph
lgraph = layerGraph(net);

% 替換最後的全連接層和分類層
numClasses = numel(categories); % 計算類別數

% 創建新層
newFcLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10); % 提高學習率以適應新類別
newClassificationLayer = classificationLayer('Name', 'new_output');

% 替換舊層
lgraph = replaceLayer(lgraph, 'fc1000', newFcLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassificationLayer);

% 設定圖像大小 (224x224x3)，並將灰度圖轉換為三通道
imds.ReadFcn = @(filename)cat(3, imresize(imread(filename), [224, 224]), ...
                                imresize(imread(filename), [224, 224]), ...
                                imresize(imread(filename), [224, 224]));

% 檢查資料集結構
disp('檢查每個類別的圖片數量:');
disp(countEachLabel(imds));

% 分割資料集為訓練集和測試集
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized'); % 70% 訓練, 30% 測試

% 設定訓練選項
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.0005, ... % 調低學習率以適應遷移學習
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

% 訓練 ResNet 模型
netTransfer = trainNetwork(imdsTrain, lgraph, options);

% 使用測試集進行預測
YPred = classify(netTransfer, imdsTest);
YTest = imdsTest.Labels;

% 計算測試集的準確率
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
