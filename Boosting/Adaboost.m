


% Weka Demo
clear;close all;clc;

javaaddpath('weka.jar');
% Add Weka path

% Load data set
data = load('X_train.csv');
labels = load('y_train.csv');
labels(labels==0) = -1;

orgdata = [data labels];

M = Fivefolds(data);

for fold = 1 : 5
    
    
    traindata = orgdata ;
    %trainlabels = labels(1:700,:);
    testdata = orgdata(M(:,fold),:);
    traindata(M(:,fold),:) = [];

    %testlabels = labels(700:801,:);
    classifiers = {};

    MAXnumofclassifiers = 30;
    Ds = ones(size(traindata,1),1)*1/size(traindata,1);
    
    for i = 1 : 1

        
     %   traindata1 = datasample(traindata,size(traindata,1),'Weights',Ds);
        
        
        
        save train.txt traindata -ascii

   %     save trainfortest.txt traindata -ascii

        ArffTrain = convertToArff('train.txt');

    %    ArffTrain4Test = convertToArff('trainfortest.txt');

        % Train a Decision Stump classifier
        classifier = weka.classifiers.meta.AdaBoostM1();
        
        classifier.setNumIterations(30);
        classifier.buildClassifier(ArffTrain);

    end

    save test.txt testdata -ascii
    ArffTest = convertToArff('test.txt');

    numInst = ArffTest.numInstances();





    for k=1:numInst
      %  ests = zeros(size(classifiers,2),1);

     %   for i = 1 : size(classifiers,2)
            ests = classifier.classifyInstance(ArffTest.instance(k-1));
     %   end

     % estimatedTestLabels(k,1) = mode(ests);
      estimatedTestLabels(k,1) = str2num(char(ArffTest.classAttribute().value((ests))));

    end

    % Compute accuracy of each fold
    testLabels = testdata(:,end);
    ACC(fold) = (sum(estimatedTestLabels == testLabels) / length(testLabels)) * 100;
    

end
%}
MeanAccuracy = mean(ACC)
Std = std(ACC)



