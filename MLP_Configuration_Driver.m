function [] = MLP_Configuration()

    % Load MNIST.
    inputValues = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');

%%add bias values to the input
    bias = 1; %wether to use bias 1 is ON 2 is OFF
    if(bias)
        inputValues = [inputValues ; ones(1,size(inputValues,2))];
    end
%%

    % Transform the labels to correct target values.
    targetValues = 0.*ones(10, size(labels, 1));
    for n = 1: size(labels, 1)
        targetValues(labels(n) + 1, n) = 1;
    end
%%
    % Input Hidden Layes Here. Eg:- For 2 layers with 10, 20 nodes input = [10 20]
    HiddenUnits = [30];
%%
    % Choose appropriate parameters. 0.001 for adam and 0.01 for adagrad
    learningRate = 0.01;
    %if GD with mometum is used
    momentum = 0.9;
    %if ADM is use
    b1 = 0.9; b2 = 0.999;epsi=10^-8;
    %methodToUse ; {1 : GD with momentum} ; {2 : Adam} ; {3 : Adagrad}
    methodToUse = 3;
%%

    % Choose activation function. Which Activation to use can be set in
    % Acitvation.m and derv_Activation.m
    activationFunction = @Activation;
    dActivationFunction = @drev_Activation;

%%
    % Choose batch size for batch update and epochs = number of iterations.
    batchSize = 100;
    epochs = 200;

    fprintf('Train perceptron with %d hidden layers.\n', length(HiddenUnits));
    fprintf('Learning rate: %d.\n', learningRate);

    Weights = trainMLP(activationFunction, dActivationFunction, methodToUse, HiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate, momentum,b1,b2,epsi,bias);

    % Load validation set.
    inputValues = loadMNISTImages('t10k-images.idx3-ubyte');
    labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    %adding bias
    if(bias)
        inputValues = [inputValues ; ones(1,size(inputValues,2))];
    end
    % Choose decision rule.
    fprintf('Testing:\n');

    [correctlyClassified, classificationErrors] = testMLP(activationFunction, Weights, inputValues, labels,bias);


    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
end
