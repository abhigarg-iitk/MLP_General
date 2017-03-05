function y = Activation(x)
% Activation function
% 
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector with activation applied.
%
% a = 1 for ReLu and a=0 for Tanh
    a  = 1;
    if a ==1
        for w = 1 : length(x)
            if(x(w)>0)
                y(w,1)=x(w);
            else
                y(w,1)=0;
            end
         end
    else
       y = 1-(2./(1 + exp(2.*x)));
end