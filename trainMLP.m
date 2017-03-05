function Weights = trainMLP(activationFunction, dActivationFunction,methodToUse, HiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate, momentum, b1, b2, epsi,bias)
% trainMLP Creates a multi-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
%
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%bias                            : Weather to use bias
%
% OUTPUT:
% Weights                        : cell of weights where Weights{i} is matrix of weights from layer i to i+1
%

% The number of training vectors.
trainingSetSize = size(inputValues, 2);

% Input vector has 784 dimensions.
inputDimensions = size(inputValues, 1);
% We have to distinguish 10 digits.
outputDimensions = size(targetValues, 1);

%Adding Bias
if(bias)
    HiddenUnits(2:end-1)=HiddenUnits(2:end-1) + 1;
end
%adding Input and output layers to hidden layers
HiddenUnits = [inputDimensions HiddenUnits outputDimensions];
noOfHiddenUnits=length(HiddenUnits);
%Weights{i} denotes weight matrix from layer i to i+1
Weights = cell(1,noOfHiddenUnits-1);
%used for Momentum
oldWeight = cell(1,noOfHiddenUnits-1);

% Initialize the weights and old weights.
for i = 1:noOfHiddenUnits-1
    Weights{i} =   2*rand(HiddenUnits(1,i+1), HiddenUnits(1,i))-1;
    Weights{i}= Weights{i}./size(Weights{i},2);
    oldWeight{i} = zeros(HiddenUnits(1,i+1), HiddenUnits(1,i));
end
%DeltaWeight will store derivative of loss function w.r.t each weight
DeltaWeight = oldWeight;
%To be used for Adam ; m : first momentum ; v = second_momentum ;
m = oldWeight;
v = oldWeight;
%To be used for Adagrad
cache = oldWeight;

figure; hold on;
for t = 1: epochs
    %use if doing batch GD
    if(batchSize>1)
        r = randperm(60000);
        batch = r(1,1:batchSize);
        %use If doing SGD
    else
        batch = floor(rand(1)*trainingSetSize + 1);
    end

    %reset all delta Weights to 0
    for layer = 1 : length(DeltaWeight)
        DeltaWeight{layer} = 0*DeltaWeight{layer};
    end

    %ActualInput{i} is Wx+b to the layer
    %ActualOutput{i} is sigma(Wx+b)
    ActualInput = cell(1,noOfHiddenUnits);
    ActualOutput = cell(1,noOfHiddenUnits);

    %Store BackProp Error
    BackPropDelta = cell(1,noOfHiddenUnits-1);

    for k = batch

        % Propagate the input vector through the network.
        % Forward Pass
        ActualInput{1} = inputValues(:, k);
        ActualOutput{1} = ActualInput{1};

        for j = 2 : noOfHiddenUnits
            ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
            ActualOutput{j} = activationFunction(ActualInput{j});
            %We don't modify bias nodes
            if(bias)
                if(j~=noOfHiddenUnits)
                    ActualOutput{j}(end) = 1;
                end
            end
        end

        %Applying Softmax
        expo = sum(exp(ActualOutput{noOfHiddenUnits}));
        for q = 1 : length(ActualOutput{noOfHiddenUnits})
            ActualOutput{noOfHiddenUnits}(q) = exp( ActualOutput{noOfHiddenUnits}(q))./expo;
        end

        targetVector = targetValues(:, k);
        %Backward Pass
        BackPropDelta{noOfHiddenUnits-1}=dActivationFunction(ActualInput{noOfHiddenUnits}).*(ActualOutput{noOfHiddenUnits} - targetVector);
        for j= noOfHiddenUnits-2:-1:1
            BackPropDelta{j} = dActivationFunction(ActualInput{j+1}).*(Weights{j+1}'*BackPropDelta{j+1});
        end

        %finally calculate delta as O(i)*BackPropDelta(j)
        for j = 1 : noOfHiddenUnits-1
            DeltaWeight{j} = DeltaWeight{j}+BackPropDelta{j}*ActualOutput{j}';
        end
    end

    if(methodToUse==2)
        for j = 1: noOfHiddenUnits-1
            DeltaWeight{j} = DeltaWeight{j}./batchSize;
            m{j} = b1.*m{j}+(1-b1).*DeltaWeight{j};
            v{j} = b2.*v{j}+(1-b2).*(DeltaWeight{j}.^2);
            mh = m{j}./(1-b1^t);
            vh = v{j}./(1-b2^t);
            Weights{j} = Weights{j} - (learningRate.*mh)./(vh.^(0.5)+epsi);
        end
    elseif methodToUse==3
        for j = 1:noOfHiddenUnits-1
            DeltaWeight{j} = DeltaWeight{j}./batchSize;
            cache{j} = cache{j} + DeltaWeight{j}.^2;
            Weights{j} = Weights{j} - learningRate.*DeltaWeight{j}./(cache{j}.^(0.5) + epsi);
        end
    else
        for j = 1 : noOfHiddenUnits-1
            DeltaWeight{j} = DeltaWeight{j}./batchSize;
            oldWeight{j} = momentum.*oldWeight{j} + learningRate.*DeltaWeight{j};
            Weights{j} = Weights{j} - oldWeight{j};
        end
    end

    %error for plotting
    error=0;
    for k = batch
        ActualInput{1} = inputValues(:, k);
        ActualOutput{1} = ActualInput{1};
        for j = 2 : noOfHiddenUnits
            ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
            ActualOutput{j} = activationFunction(ActualInput{j});
            %We don't modify bias nodes
            if(bias)
                if(j~=noOfHiddenUnits)
                    ActualOutput{j}(end) = 1;
                end
            end
        end

        %Applying Softmax
        expo = sum(exp(ActualOutput{noOfHiddenUnits}));
        for q = 1 : length(ActualOutput{noOfHiddenUnits})
            ActualOutput{noOfHiddenUnits}(q) = exp( ActualOutput{noOfHiddenUnits}(q))./expo;
        end

        targetVector = targetValues(:, k);

        error=error+(0.5)*norm(ActualOutput{noOfHiddenUnits} - targetVector);
    end
    error = error/batchSize;
    plot(t,error,'*');
    drawnow;
end;


%%
%Below is the code for Computing numeric Gradient
%{
    epi = .0001;
    two_epi = Weights;
    real_epi = Weights;


    for layer = 1 : length(DeltaWeight)
            DeltaWeight{layer} = 0*DeltaWeight{layer};
       end

            % Propagate the input vector through the network.
            ActualInput = cell(1,noOfHiddenUnits);
            ActualOutput = cell(1,noOfHiddenUnits);
            ActualInput{1} = inputValues(:, k);
            ActualOutput{1} = ActualInput{1};
            for j = 2 : noOfHiddenUnits
                ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
                ActualOutput{j} = activationFunction(ActualInput{j});
                %if(j~=noOfHiddenUnits)
                %    ActualOutput{j}(end) = 1;
                %end
            end

            expo = sum(exp(ActualOutput{noOfHiddenUnits}));
            for q = 1 : length(ActualOutput{noOfHiddenUnits})
                ActualOutput{noOfHiddenUnits}(q) = exp( ActualOutput{noOfHiddenUnits}(q))./expo;
            end
            targetVector = targetValues(:, k);

            BackPropDelta = cell(1,noOfHiddenUnits-1);
            % Backpropagate the errors.

            BackPropDelta{noOfHiddenUnits-1}=dActivationFunction(ActualInput{noOfHiddenUnits}).*(ActualOutput{noOfHiddenUnits} - targetVector);
            for j= noOfHiddenUnits-2:-1:1
                BackPropDelta{j} = dActivationFunction(ActualInput{j+1}).*(Weights{j+1}'*BackPropDelta{j+1});
            end

            for j = 1 : noOfHiddenUnits-1
                DeltaWeight{j} = DeltaWeight{j}+BackPropDelta{j}*ActualOutput{j}';
            end

    for qq = 1:2
        for ii = 1 : size(Weights{qq},1)
            for jj = 1:size(Weights{qq},2)
                Weights{qq}(ii,jj)= Weights{qq}(ii,jj)+epi;

            % Propagate the input vector through the network.
            ActualInput = cell(1,noOfHiddenUnits);
            ActualOutput = cell(1,noOfHiddenUnits);
            ActualInput{1} = inputValues(:, k);
            ActualOutput{1} = ActualInput{1};
            for j = 2 : noOfHiddenUnits
                ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
                ActualOutput{j} = activationFunction(ActualInput{j});
                %if(j~=noOfHiddenUnits)
                %    ActualOutput{j}(end) = 1;
                %end
            end

            expo = sum(exp(ActualOutput{noOfHiddenUnits}));
            for q = 1 : length(ActualOutput{noOfHiddenUnits})
                ActualOutput{noOfHiddenUnits}(q) = exp( ActualOutput{noOfHiddenUnits}(q))./expo;
            end
            targetVector = targetValues(:, k);

            r=norm(ActualOutput{noOfHiddenUnits} - targetVector);

              Weights{qq}(ii,jj)= Weights{qq}(ii,jj)-(2.*epi);

            % Propagate the input vector through the network.
            ActualInput = cell(1,noOfHiddenUnits);
            ActualOutput = cell(1,noOfHiddenUnits);
            ActualInput{1} = inputValues(:, k);
            ActualOutput{1} = ActualInput{1};
            for j = 2 : noOfHiddenUnits
                ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
                ActualOutput{j} = activationFunction(ActualInput{j});
                %if(j~=noOfHiddenUnits)
                %    ActualOutput{j}(end) = 1;
                %end
            end

            expo = sum(exp(ActualOutput{noOfHiddenUnits}));
            for q = 1 : length(ActualOutput{noOfHiddenUnits})
                ActualOutput{noOfHiddenUnits}(q) = exp( ActualOutput{noOfHiddenUnits}(q))./expo;
            end
            targetVector = targetValues(:, k);

            r=r - norm((ActualOutput{noOfHiddenUnits} - targetVector));

            two_epi{qq}(ii,jj) = r./(2*epi);

            Weights{qq}(ii,jj)= Weights{qq}(ii,jj)+(epi);


            end
        end
    end
%}


%%
%Predicting validation Accuracy
%{
%inputValues = loadMNISTImages('t10k-images.idx3-ubyte');
%labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
%inputValues = [inputValues ; ones(1,size(inputValues,2))];

testSetSize = size(inputValues, 2);
classificationErrors = 0;
correctlyClassified = 0;

for n = 1: testSetSize
    ActualInput = cell(1,noOfHiddenUnits);
    ActualOutput = cell(1,noOfHiddenUnits);
    %inputVector = inputValues(:, n(k));
    ActualInput{1} = inputValues(:, n);
    ActualOutput{1} = ActualInput{1};
    for j = 2 : noOfHiddenUnits
        ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
        ActualOutput{j} = activationFunction(ActualInput{j});
        if(j~=noOfHiddenUnits)
            ActualOutput{j}(end) = 1;
        end
    end
    %inputVector = inputValues(:, n);
    %outputVector = evaluateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputVector);
    [m class] = max(ActualOutput{noOfHiddenUnits});
    %class = decisionRule(outputVector);
    if class == labels(n) + 1
        correctlyClassified = correctlyClassified + 1;
    else
        classificationErrors = classificationErrors + 1;
    end;
end;
correctlyClassified = correctlyClassified./10000;
%}
end
