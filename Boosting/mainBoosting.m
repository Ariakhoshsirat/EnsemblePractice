


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
    Ds = ones(size(traindata,1),1)* (1/size(traindata,1));
    
    for i = 1 : MAXnumofclassifiers

        %inds = randi(size(traindata,1),size(traindata,1),1);
        %traindata1 = traindata(inds,:);
        
        traindata1 = datasample(traindata,size(traindata,1),'Weights',Ds);
        

        
        trainlabels = traindata1(:,end);
        if(size(unique(trainlabels),1)==1)
            continue;
        end
        
        save train.txt traindata1 -ascii

        save trainfortest.txt traindata -ascii

        ArffTrain = convertToArff('train.txt');

        ArffTrain4Test = convertToArff('trainfortest.txt');

        % Train a Decision Stump classifier
        classifier = weka.classifiers.trees.DecisionStump();


        classifier.buildClassifier(ArffTrain);

        classifiers{i} = classifier;
        trainlabelsss = traindata(:,end);
        
        
        numInsttss = ArffTrain4Test.numInstances();
        
        estlabelstrain = zeros(numInsttss,1);
        
        
        
        for k=1:numInsttss
            estimated = classifiers{i}.classifyInstance(ArffTrain4Test.instance(k-1));
            estlabelstrain(k) = str2num(char(ArffTrain4Test.classAttribute().value((estimated))));
        end
        negs = find(estlabelstrain ~= trainlabelsss);
        corrects = find(estlabelstrain == trainlabelsss);
        
        E = sum(Ds(negs));
        if E == 0
            weights(i) = 0;
            break;
        end
        Bt = E / (1-E) ;  %1/2 * log((1-E)/E);
      % Bt = real(at);
      % Weights = 
        Ds(corrects) = Ds(corrects) * Bt;
        
        Ds = Ds ./ sum(Ds);
        weights(i) = log(1/Bt);
        
%       Ds = normc(Ds);
        
    end

    save test.txt testdata -ascii
    ArffTest = convertToArff('test.txt');

    numInst = ArffTest.numInstances();





    for k=1:numInst
        ests = zeros(size(classifiers,2),1);

        for i = 1 : size(classifiers,2)
            ests(i) = classifiers{i}.classifyInstance(ArffTest.instance(k-1));
            ests(i) = str2num(char(ArffTest.classAttribute().value((ests(i) ))));
        end

     % estimatedTestLabels(k,1) = mode(ests);
      estimatedTestLabels(k,1) = sign(weights*ests);

    end

    % Compute accuracy of each fold
    testLabels = testdata(:,end);
    ACC(fold) = (sum(estimatedTestLabels == testLabels) / length(testLabels)) * 100;
    

end
%}
MeanAccuracy = mean(ACC)
Std = std(ACC)



