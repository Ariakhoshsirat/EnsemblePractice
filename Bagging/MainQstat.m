


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

    numofclassifiers = 5;

    for i = 1 : numofclassifiers

        inds = randi(size(traindata,1),size(traindata,1),1);
        traindata1 = traindata(inds,:);


        save train.txt traindata1 -ascii



        ArffTrain = convertToArff('train.txt');



        % Train a J48 classifier
        classifier = weka.classifiers.trees.J48();


        classifier.buildClassifier(ArffTrain);

        classifiers{i} = classifier;

    end

    save test.txt testdata -ascii
    ArffTest = convertToArff('test.txt');

    numInst = ArffTest.numInstances();



    testLabels = testdata(:,end);
    
    for i = 1 : numofclassifiers
        estlabels = zeros(numInst,1);
        for k=1:numInst  
               % ests(i) = classifiers{i}.classifyInstance(ArffTest.instance(k-1));
               estlabels(k) = classifiers{i}.classifyInstance(ArffTest.instance(k-1));
               % estimatedTestLabels(k,1) = str2num(char(ArffTest.classAttribute().value((mode(ests)))));
               Pos{i} = find(estlabels == testLabels);
               Neg{i} = find(estlabels ~= testLabels);
        end
    end
    Q = 0;
    for i = 1 : numofclassifiers-1
        for j = i+1 : numofclassifiers
            
            n11 = size(intersect(Pos{i},Pos{j}),1);
            n10 = size(intersect(Pos{i},Neg{j}),1);
            n01 = size(intersect(Neg{i},Pos{j}),1);
            n00 = size(intersect(Neg{i},Neg{j}),1);
            Q = Q + ((n11*n00) - (n10*n01))/((n11*n00) + (n10*n01));
        end
    end
    
    Q = Q*2/(numofclassifiers*(numofclassifiers-1));
    QS(fold) = Q;
    % Compute accuracy of each fold
    
  %  ACC(fold) = (sum(estimatedTestLabels == testLabels) / length(testLabels)) * 100;
    

end
%}
MeanQ = mean(QS)
StdQ = std(QS)



