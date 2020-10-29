


% Weka Demo
clear;close all;clc;

javaaddpath('weka.jar');
% Add Weka path

% Load data set
data = load('X_train.csv');
labels = load('y_train.csv');

orgdata = [data labels];

M = Fivefolds(data);

for fold = 1 : 5
    
    
    traindata = orgdata ;
    %trainlabels = labels(1:700,:);
    testdata = orgdata(M(:,fold),:);
    traindata(M(:,fold),:) = [];

    %testlabels = labels(700:801,:);
    classifiers = {};

    numofclassifiers = 1;

    for i = 1 : numofclassifiers

     %   inds = randi(size(traindata,1),size(traindata,1),1);
     %   traindata1 = traindata(inds,:);


        save train.txt traindata -ascii



        ArffTrain = convertToArff('train.txt');



        % Train a  classifier
        classifier = weka.classifiers.trees.RandomForest();
        
        classifier.setMaxDepth(30);
        classifier.setNumFeatures(5);
        classifier.setNumTrees(2);
        


        classifier.buildClassifier(ArffTrain);
        
        classifiers{i} = classifier;
    end

    
    save test.txt testdata -ascii
    ArffTest = convertToArff('test.txt');
    %-------------------Train Acc------------------
    numInst = ArffTrain.numInstances();

    for k=1:numInst
        ests = zeros(numofclassifiers,1);

        for i = 1 : numofclassifiers
            ests = classifiers{i}.classifyInstance(ArffTrain.instance(k-1));
        end

      estimatedTestLabelstrain(k,1) = ests;
     %  estimatedTestLabels(k,1) = str2num(char(ArffTest.classAttribute().value((mode(ests)))));

    end

    % Compute accuracy of each fold
    trainlabels = traindata(:,end);
    ACCTrain(fold) = (sum(estimatedTestLabelstrain == trainlabels) / length(trainlabels)) * 100;
    
    
    
    
    
    
    
    
    
    %--------------------------------------
    numInst = ArffTest.numInstances();

    for k=1:numInst
        ests = zeros(numofclassifiers,1);

        for i = 1 : numofclassifiers
            ests = classifiers{i}.classifyInstance(ArffTest.instance(k-1));
        end

      estimatedTestLabels(k,1) = ests;
     %  estimatedTestLabels(k,1) = str2num(char(ArffTest.classAttribute().value((mode(ests)))));

    end

    % Compute accuracy of each fold
    testLabels = testdata(:,end);
    ACC(fold) = (sum(estimatedTestLabels == testLabels) / length(testLabels)) * 100;
    
    
end
%}
MeanAccuracy = mean(ACC)
Std = std(ACC)

MeanAccuracyTrain = mean(ACCTrain)
StdTrain = std(ACCTrain)


