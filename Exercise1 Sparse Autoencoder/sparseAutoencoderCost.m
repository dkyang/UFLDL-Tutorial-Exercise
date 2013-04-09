function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
[ndims, m] = size(data);


z2 = zeros(hiddenSize, m);
z3 = zeros(visibleSize, m);
a1 = zeros(ndims, m);
a2 = zeros(size(z2));
a3 = zeros(size(z3));
%autoencode use inputs as target values
y  = zeros(ndims, m);

a1 = data;
y = data;

deltaW1 = zeros(size(W1));
deltab1 = zeros(size(b1));
JW1grad = zeros(size(W1));
Jb1grad = zeros(size(b1));
deltaW2 = zeros(size(W2));
deltab2 = zeros(size(b2));
JW2grad = zeros(size(W2));
Jb2grad = zeros(size(b2));

%forward pass 
for i = 1 : m
    z2(:,i) = W1 * data(:,i) + b1;
    a2(:,i) = sigmoid(z2(:,i));
    z3(:,i) = W2 * a2(:,i) + b2;
    a3(:,i) = sigmoid(z3(:,i));
end

rho = zeros(hiddenSize, 1);
rho = (1. / m) * sum(a2, 2);
sp = sparsityParam;

for i = 1 : m
    delta3 = -(y(:,i) - a3(:,i)) .* sigmoidGrad(z3(:,i));
    delta2 = ( W2' * delta3 + beta * (-sp ./ rho + (1-sparsityParam) ./ (1-rho) ) ) ...
        .* sigmoidGrad(z2(:,i));
    
    JW1grad = delta2 * a1(:,i)';
    Jb1grad = delta2;
    JW2grad = delta3 * a2(:,i)';
    Jb2grad = delta3;

    deltaW1 = deltaW1 + JW1grad;
    deltab1 = deltab1 + Jb1grad;
    deltaW2 = deltaW2 + JW2grad;
    deltab2 = deltab2 + Jb2grad;
end


W1grad = (1. / m) * deltaW1 + lambda * W1;
b1grad = (1. / m) * deltab1;
W2grad = (1. / m) * deltaW2 + lambda * W2;
b2grad = (1. / m) * deltab2;

cost = (1. / m) * sum((1. / 2) * sum((a3 - y).^2)) + ...
    (lambda / 2.) * (sum(sum(W1.^2)) + sum(sum(W2.^2))) + ...
    beta * sum( sp*log(sp./rho) + (1-sp)*log((1-sp)./(1-rho)) );


function grad = sigmoidGrad(x)
    e_x = exp(-x);
    grad = e_x ./ ((1 + e_x).^2); 
end



%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

