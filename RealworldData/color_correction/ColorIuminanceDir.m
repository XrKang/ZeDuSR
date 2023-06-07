clear;

addpath(genpath('../Data/'));
addpath(genpath('./Opt_reg_luminance/'));
addpath(genpath('./Opt_reg_color/'));
addpath(genpath('./Opt_reg/'));

ref_folder  = '../Data//WideView_crop_bic/';
input_folder  = '../Data//TeleView_SIFTAlign/';
save_folder = '../Data//TeleView_SIFTAlign_cor/';



% filepaths = dir(fullfile(ref_folder, '*.png')); % iPhone12
filepaths = dir(fullfile(ref_folder, '*.jpg')); % iPhone11

for i = 1:1:size(filepaths)
    path1 = fullfile(ref_folder, filepaths(i).name);       % reference image
    path2 = fullfile(input_folder, filepaths(i).name);       % reference image

    
    I1  = im2double(imread(path1));       % reference image
    I2  = im2double(imread(path2));       % target image

    [I2_l] = luminance_transfer(I1,I2);              % transfer luminance
    [I2_c] = color_transfer(I1,I2_l);                % transfer color
    imwrite(I2_c, [save_folder, filepaths(i).name])
end
