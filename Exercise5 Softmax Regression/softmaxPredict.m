function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
numClasses = softmaxModel.numClasses;
inputSize = softmaxModel.inputSize;
theta = reshape(theta, numClasses, inputSize);

M = exp(theta * data);
%可以用这种方法替换我前面写的repmat
M = bsxfun(@rdivide, M, sum(M));

[p,pred] = max(M, [], 1);





% ---------------------------------------------------------------------

end

