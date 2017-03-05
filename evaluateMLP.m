function output = evaluateMLP(activationFunction,Weights, Sample,bias)
% Evaluates the MLP
%INPUT :- activationFunction, Weights :- weight matrix ; Sample :- input
%example on which to evauate
%OUTPUT :- output final result vector after passing through network

    %Forward Pass
    noOfHiddenUnits = length(Weights)+1;
    ActualInput = cell(1,noOfHiddenUnits);
    ActualOutput = cell(1,noOfHiddenUnits);
    %inputVector = inputValues(:, n(k));
    ActualInput{1} = Sample;
    %ActualInput{1} = inputValues(:, n);
    ActualOutput{1} = ActualInput{1};
    for j = 2 : noOfHiddenUnits
        ActualInput{j} = Weights{j-1}*ActualOutput{j-1};
        ActualOutput{j} = activationFunction(ActualInput{j});
        if(bias)
            if(j~=noOfHiddenUnits)
                ActualOutput{j}(end) = 1;
            end
        end
    end
    output = ActualOutput{noOfHiddenUnits};

end