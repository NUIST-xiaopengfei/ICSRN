clear;
clc;

% gpu
gpu_id = 0;
caffe.set_mode_gpu(); % for GPU
caffe.set_device(gpu_id);

weights_ICSRN = 'ICSRN.caffemodel';
model_ICSRN = 'ICSRN_mat.prototxt';
img  = imread('FY_2G_test_15/14.bmp');
img = im2double(img);
size_o = size(img);

%% add stripe noise
 img_n = img + repmat(random('norm', 0, 0.02, 1, size_o(2)), size_o(1), 1);

 %% caffe Forward calculation
 net = caffe.Net(model_ICSRN, weights_ICSRN, 'test');
 data = permute(img_n,[2, 1, 3]);
 res = net.forward({data});
 outputdata = res{1}; 
 outputdata=outputdata';
 
figure,imshow(img_n);title('Striped image');
figure,imshow(outputdata);title('ICSRN Reconstruction');