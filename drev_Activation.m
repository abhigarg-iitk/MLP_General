function y = drev_Activation(x)
% Derivative of the Activation Function.
%
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector where the derivative of the Activation function was
% applied element by element.
%a = 1 for ReLu and a=0 for Tanh
%%
    a=1;
    if a==0
        y = 1-Activation(x).^2;
    else
        for w = 1 : length(x)
            if(x(w)>0)
                y(w,1)=1;
            else
                y(w,1)=0;
            end
        end
    end
end