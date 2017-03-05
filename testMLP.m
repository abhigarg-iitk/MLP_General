function [correctlyClassified, classificationErrors] = validateMLP(activationFunction, Weights, inputValues, labels,bias)
% Validate the MLP using the
% validation set.
%
% INPUT:
% activationFunction             : Activation function to be used
% Weights                        : Weights of the Layers
% inputValues                    : Input values for training (784 x 10000).
% labels                         : Labels for validation (1 x 10000).
% bias                           : Weather to use bias
%
% OUTPUT:
% correctlyClassified            : Number of correctly classified values.
% classificationErrors           : Number of classification errors.
%

    testSetSize = size(inputValues, 2);
    classificationErrors = 0;
    correctlyClassified = 0;

    for n = 1: testSetSize
        inputVector = inputValues(:, n);
        outputVector = evaluateMLP(activationFunction, Weights, inputVector, bias);

        [m class] = max(outputVector);
        %class = decisionRule(outputVector);
        if class == labels(n) + 1
            correctlyClassified = correctlyClassified + 1;
        else
            classificationErrors = classificationErrors + 1;
        end;
    end
end
