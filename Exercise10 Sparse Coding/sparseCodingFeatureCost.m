function [cost, grad] = sparseCodingFeatureCost(weightMatrix, featureMatrix, visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingFeatureCost - given the weights in weightMatrix,
%                          computes the cost and gradient with respect to
%                          the features, given in featureMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.

    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2);

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);

    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   features given in featureMatrix.     
    %   You may wish to write the non-topographic version, ignoring
    %   the grouping matrix groupMatrix first, and extend the 
    %   non-topographic version to the topographic version later.
    % -------------------- YOUR CODE HERE --------------------
    
    %% topographic
     k = 1;
%      lamda = 0;
%      gamma = 0;
     T = weightMatrix * featureMatrix - patches;
     cost = k * trace(T' * T) + ...
         lambda * sum(sum(sqrt(groupMatrix * featureMatrix.^2 + epsilon))) + ...
         gamma * trace(weightMatrix' * weightMatrix);
%       grad =  k * 2 * weightMatrix' * T + ...
%           lambda * featureMatrix ./ sqrt(featureMatrix.^2 + epsilon);
     grad =  k * 2 * weightMatrix' * T + ...
          lambda * groupMatrix * featureMatrix ./ sqrt(groupMatrix * featureMatrix.^2 + epsilon);
     %grad的第一项见公式108
     %我不知道自己的思路对不对，第二项是featureMatrix每个样本的项的和。
%      对每个样本而言，展开的形式是sqrt(s1^2+epsilon) + sqrt(s2^2+epsilon) + sqrt(sk^2+epsilon);
%     每列的元素都是上式对sk求导。 每个样本（featureMatrix的列向量）都如此处理，得到下式的第二项
  
   
     
%      %% non-topographic
%      k = 1;
%      T = weightMatrix * featureMatrix - patches;
%      cost = k * trace(T' * T) + ...
%          lambda * sum(sum(sqrt(featureMatrix.^2 + epsilon))) + ...
%          gamma * trace(weightMatrix' * weightMatrix);
%      
%     grad =  k * 2 * weightMatrix' * T + ...
%      lambda * featureMatrix ./ sqrt(featureMatrix.^2 + epsilon);
     grad = grad(:);
end