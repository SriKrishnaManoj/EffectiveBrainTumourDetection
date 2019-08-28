function ObjFcn = makeObjFcn(imdsTrain,imdsValidation,datasource,imageSize,i)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        numClasses = numel(unique(imdsTrain.Labels));
        initialNumFilters = 16;
        
        layers = [
            imageInputLayer(imageSize)
            
            convBlock(3,initialNumFilters)
            maxPooling2dLayer(2,'Stride',2)
            
            convBlock(3,2*initialNumFilters)
            maxPooling2dLayer(2,'Stride',2)
            
            convBlock(3,4*initialNumFilters)
            maxPooling2dLayer(8)
            
            convBlock(3,16*initialNumFilters)
            averagePooling2dLayer(8)
            
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
        
        miniBatchSize = 128;
        validationFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
        options = trainingOptions('adam',...
            'InitialLearnRate',optVars.InitialLearnRate,...
            'GradientDecayFactor',optVars.GradientDecayFactor, ...
            'SquaredGradientDecayFactor',optVars.SquaredGradientDecayFactor, ...
            'MaxEpochs',optVars.MaxEpochs, ...
            'LearnRateSchedule','piecewise',...
            'LearnRateDropPeriod',35,...
            'LearnRateDropFactor',0.1,...
            'MiniBatchSize',miniBatchSize,...
            'L2Regularization',optVars.L2Regularization,...
            'Shuffle','every-epoch',...
            'Verbose',false,...
            'Plots','training-progress',...
            'ValidationData',imdsValidation,...
            'ValidationPatience',Inf,...
            'ValidationFrequency',validationFrequency,...
            'OutputFcn',@(info)savetrainingplot(info, i));
        
        trainedNet = trainNetwork(datasource,layers,options);
        
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))        
        rootfile = '../Network Progress - 4/';
        YPredicted = classify(trainedNet,imdsValidation);
        valError = 1 - mean(YPredicted == imdsValidation.Labels);
        fileName = strcat(num2str(valError),'-',num2str(i),'.mat');
        save(strcat(rootfile,'Trained Nets/',fileName),'trainedNet','valError','options')
        cons = [];
        i=i+1;
    end
end

function layers = convBlock(filterSize,numFilters)
layers = [
    convolution2dLayer(filterSize,numFilters,'Padding','same')
    batchNormalizationLayer
    reluLayer];
end

function stop=savetrainingplot(info, i)
    stop=false;  %prevents this function from ending trainNetwork prematurely
    if info.State=='done'   %check if all iterations have completed
    % if true
            rootfile = '../Network Progress - 4/';
            fileName = strcat(num2str(info.ValidationAccuracy),'-',num2str(i),'.jpg');
            saveas(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'),strcat(rootfile,'Trained Plots/',fileName))  % save figure as .jpg
    end
end