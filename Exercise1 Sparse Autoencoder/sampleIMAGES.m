function patches = sampleIMAGES()

% sampleIMAGES
% Returns 10000 patches for training

load IMAGES;    % load images from disk 

patchsize = 8;  % we'll use 8x8 patches 
numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

%size of images
height = 512;
width = 512;

%indices of randomly picked images 
img_ind = unidrnd(10, [1, numpatches]); 

%row and column indices of randomly picked images
row_ind = unidrnd(height - patchsize + 1, [1, numpatches]);
col_ind = unidrnd(width - patchsize + 1, [1, numpatches]);

%length of unrolled patches
patch_len = patchsize * patchsize;

for i = 1 : numpatches
    cur_patch = IMAGES(row_ind(i) : row_ind(i)+patchsize-1,...
                       col_ind(i) : col_ind(i)+patchsize-1,...
                       img_ind(i));
    patches(:,i) = reshape(cur_patch, [patch_len, 1]);
end

% %这种写法完全是错误的！row_ind : row_ind+patchsize-1只取第一个row_ind 
% % get patches haven't been unrolled
% blocks = IMAGES(row_ind : row_ind+patchsize-1,...
%                 col_ind : col_ind+patchsize-1,...
%                 img_ind);
%             
% % convert each image patch into a 64-dimensional vector
% patches = reshape(blocks, [patch_len, numpatches]);




%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
