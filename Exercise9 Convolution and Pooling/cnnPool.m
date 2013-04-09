function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------
poolLen = floor(convolvedDim / poolDim);
rb = 0;
re = 0;
cb = 0;
ce = 0;

for i = 1 : numFeatures
    for j = 1 : numImages
        for r = 1 : poolLen
            for c = 1 : poolLen
                rb = 1 + poolDim * (r-1);
                re = poolDim * r;
                cb = 1 + poolDim * (c-1);
                ce = poolDim * c;
%                 blockFeatures = convolvedFeatures(i, j, rb : re, cb : ce);
                pooledFeatures(i, j, r, c) = ...
                    mean(mean(convolvedFeatures(i, j, rb : re, cb : ce)));
            end
        end
    end
end

end
