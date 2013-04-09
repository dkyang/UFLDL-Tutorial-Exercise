function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

%这个函数就是建立矩阵，以labels为横坐标，1:numCases为纵坐标的位置值为1，其他位置值为0.
% 矩阵为M*N，M = max(labels); N = numCases
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


M = theta * data;

p = exp(M) ./ repmat(sum(exp(M)), numClasses, 1);
cost = -(1. / numCases) * sum(sum(groundTruth .* log(p))) + (lambda / 2.) * sum(sum(theta.^2));
thetagrad = -(1. / numCases) * (groundTruth - p) * data' + lambda * theta;




% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

