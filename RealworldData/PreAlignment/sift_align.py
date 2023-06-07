from __future__ import print_function

import cv2
import numpy as np
import glob
import os
import argparse
from PIL import Image

def modcrop(im, modulo):
	if len(im.shape) == 3:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1], :]
	elif len(im.shape) == 2:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1]]
	return im


def shave(im, border):
	return im[border[0] : -border[0], 
		      border[1] : -border[1], :]

def alignImages(im1, im2, VERBOSE=False):
	GOOD_MATCH_PERCENT = 0.1
	
	# Convert images to grayscale
	# im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	# im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	im1Gray = im1
	im2Gray = im2
	# Detect SIFT features and compute descriptors
	sift = cv2.SIFT_create()
	keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)
	
	# Match features
	if descriptors1 is None or descriptors2 is None:
		return False
	
	matcher = cv2.DescriptorMatcher_create('BruteForce')
	matches = matcher.match(descriptors1, descriptors2, None)
	
	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False)
	
	# Remove not so good matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	# numGoodMatches = int(np.maximum(np.minimum(len(keypoints1), len(keypoints2)) * GOOD_MATCH_PERCENT, 100))
	#numGoodMatches = int(np.minimum(len(keypoints1), len(keypoints2)))
	matches = matches[:numGoodMatches]

	# if VERBOSE:
	# 	print('Key points in input image: %d' % len(keypoints1))
	# 	print('Key points in ref image  : %d' % len(keypoints2))
	# 	print('Matched points           : %d' % len(matches))
	
	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)
	
	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt
	
	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
	
	if h is None:
		return False
	
	# if VERBOSE:
	# 	print('Estimated homography: \n',  h)
	
	# Use homography
	height, width, _ = im2.shape
	im1Reg = cv2.warpPerspective(im1, h, (width, height), flags=cv2.INTER_CUBIC)
	
	return im1Reg

# ------------------------------------------Real Alignment------------------------------------------------

def sift_align_ultra2tele(path_hr, path_lr, name, save_path_tele2ultra, save_path_ultraCrop, save_path_ultraCropBic):
	print(path_hr, path_lr)

	im_hr = cv2.imread(path_hr)
	im_lr = cv2.imread(path_lr)

	lh, lw, c = im_lr.shape
	im_lr = im_lr[lh // 10 * 4:lh // 10 * 6, lw // 10 * 4:lw // 10 * 6, :]
	lh, lw, c = im_lr.shape
	im_lr_bic = cv2.resize(im_lr, (int(lw * 4), int(lh * 4)), interpolation=cv2.INTER_CUBIC)
	im_hr_down = cv2.resize(im_hr, (int(lw * 4), int(lh * 4)), interpolation=cv2.INTER_CUBIC)

	align_hr = alignImages(im_hr_down, im_lr_bic, VERBOSE=True)

	cv2.imwrite(os.path.join(save_path_tele2ultra, name), align_hr)
	cv2.imwrite(os.path.join(save_path_ultraCrop, name), im_lr)
	cv2.imwrite(os.path.join(save_path_ultraCropBic, name), im_lr_bic)

def align_iphone12_real_utlra2tele(args):
	print("iPhone12 Ultra SR by using Tele")

	ultra_dir = args.wide_dir
	tele_dir = args.tele_dir
	filenames = os.listdir(tele_dir)

	save_path_tele2ultra = args.Tele_savePath  # ultra2tele/tele_SIFTAlign
	save_path_ultraCrop = args.WideCrop_savePath  # ultra2tele/ultra_crop
	save_path_ultraCropBic = args.WideCrop_savePath + '_bic'  # ultra2tele/ultra_crop_bic

	if not os.path.exists(save_path_tele2ultra):
		os.makedirs(save_path_tele2ultra)
	if not os.path.exists(save_path_ultraCrop):
		os.makedirs(save_path_ultraCrop)
	if not os.path.exists(save_path_ultraCropBic):
		os.makedirs(save_path_ultraCropBic)

	for filename in filenames:
		ultra_img = os.path.join(ultra_dir, filename)
		tele_img = os.path.join(tele_dir, filename)
		sift_align_ultra2tele(tele_img, ultra_img, filename, save_path_tele2ultra, save_path_ultraCrop, save_path_ultraCropBic)

# ------------------------------------------------------------------------------------------

def sift_align_wide2tele(path_hr, path_lr, name, save_path_tele2wide, save_path_wideCrop, save_path_wideCropBic):
	print(path_hr, path_lr)

	im_hr = cv2.imread(path_hr)
	im_lr = cv2.imread(path_lr)

	lh, lw, c = im_lr.shape
	im_lr = im_lr[lh // 10 * 3:lh // 10 * 7, lw // 10 * 3:lw // 10 * 7, :]
	lh, lw, c = im_lr.shape
	im_lr_bic = cv2.resize(im_lr, (int(lw * 2), int(lh * 2)), interpolation=cv2.INTER_CUBIC)
	im_hr_down = cv2.resize(im_hr, (int(lw * 2), int(lh * 2)), interpolation=cv2.INTER_CUBIC)

	align_hr = alignImages(im_hr_down, im_lr_bic, VERBOSE=True)

	cv2.imwrite(os.path.join(save_path_tele2wide, name), align_hr)
	cv2.imwrite(os.path.join(save_path_wideCrop, name), im_lr)
	cv2.imwrite(os.path.join(save_path_wideCropBic, name), im_lr_bic)

def align_iphone12_real_wide2tele(args):
	print("iPhone12 Wide SR by using Tele")

	wide_dir = args.wide_dir
	tele_dir = args.tele_dir

	filenames = os.listdir(tele_dir)

	# ----------------------------------------------------------------------------------------------
	save_path_tele2wide = args.Tele_savePath
	if not os.path.exists(save_path_tele2wide):
		os.makedirs(save_path_tele2wide)

	save_path_wideCrop = args.WideCrop_savePath
	if not os.path.exists(save_path_wideCrop):
		os.makedirs(save_path_wideCrop)

	save_path_wideCropBic = args.WideCrop_savePath + '_bic'
	if not os.path.exists(save_path_wideCropBic):
		os.makedirs(save_path_wideCropBic)
	# ----------------------------------------------------------------------------------------------

	for filename in filenames:
		wide_img = os.path.join(wide_dir, filename)
		tele_img = os.path.join(tele_dir, filename)
		sift_align_wide2tele(tele_img, wide_img, filename, save_path_tele2wide, save_path_wideCrop, save_path_wideCropBic)

# ------------------------------------------------------------------------------------------

def sift_align_ultra2wide(path_hr, path_lr, name, save_path_wide2ultra, save_path_ultraCrop, save_path_ultraCropBic):
	print(path_hr, path_lr)

	im_hr = cv2.imread(path_hr)
	im_lr = cv2.imread(path_lr)

	lh, lw, c = im_lr.shape
	im_lr = im_lr[lh // 4 * 1:lh // 4 * 3, lw // 4 * 1:lw // 4 * 3, :]
	lh, lw, c = im_lr.shape
	im_lr_bic = cv2.resize(im_lr, (int(lw * 2), int(lh * 2)), interpolation=cv2.INTER_CUBIC)
	im_hr_down = cv2.resize(im_hr, (int(lw * 2), int(lh * 2)), interpolation=cv2.INTER_CUBIC)

	align_hr = alignImages(im_hr_down, im_lr_bic, VERBOSE=True)

	cv2.imwrite(os.path.join(save_path_wide2ultra, name), align_hr)
	cv2.imwrite(os.path.join(save_path_ultraCrop, name), im_lr)
	cv2.imwrite(os.path.join(save_path_ultraCropBic, name), im_lr_bic)


def align_iphone12_real_ultra2wide(args):
	# print("iPhone12 Ultra SR by using Wide")
	ultra_dir = args.wide_dir
	wide_dir = args.tele_dir

	filenames = os.listdir(wide_dir)

	# ----------------------------------------------------------------------------------------------
	save_path_wide2ultra = args.Tele_savePath  # ultra2wide/wide_SIFTAlign
	if not os.path.exists(save_path_wide2ultra):
		os.makedirs(save_path_wide2ultra)

	save_path_ultraCrop_2wide = args.WideCrop_savePath  # ultra2wide/ultra_crop
	if not os.path.exists(save_path_ultraCrop_2wide):
		os.makedirs(save_path_ultraCrop_2wide)

	save_path_ultraCropBic_2wide = args.WideCrop_savePath + '_bic'  # ultra2wide/ultra_crop_bic
	if not os.path.exists(save_path_ultraCropBic_2wide):
		os.makedirs(save_path_ultraCropBic_2wide)
	# ----------------------------------------------------------------------------------------------

	for filename in filenames:
		ultra_img = os.path.join(ultra_dir, filename)
		wide_img = os.path.join(wide_dir, filename)
		sift_align_ultra2wide(wide_img, ultra_img, filename, save_path_wide2ultra, save_path_ultraCrop_2wide, save_path_ultraCropBic_2wide)

# ------------------------------------------------------------------------------------------

def sift_align_wide2tele_iphone11(path_hr, path_lr, name, save_path_tele2wide, save_path_wideCrop, save_path_wideCropBic):
	print(path_hr, path_lr)

	im_hr = cv2.imread(path_hr)
	im_lr = cv2.imread(path_lr)

	lh, lw, c = im_lr.shape
	im_lr = im_lr[lh // 10 * 3:lh // 10 * 7, lw // 10 * 3:lw // 10 * 7, :]
	lh, lw, c = im_lr.shape
	im_lr_bic = cv2.resize(im_lr, (int(lw * 2), int(lh * 2)), interpolation=cv2.INTER_CUBIC)
	im_hr_down = cv2.resize(im_hr, (int(lw * 2), int(lh * 2)), interpolation=cv2.INTER_CUBIC)

	align_hr = alignImages(im_hr_down, im_lr_bic, VERBOSE=True)

	cv2.imwrite(os.path.join(save_path_tele2wide, name), align_hr)
	cv2.imwrite(os.path.join(save_path_wideCrop, name), im_lr)
	cv2.imwrite(os.path.join(save_path_wideCropBic, name), im_lr_bic)

def align_iphone11_real_wide2tele(args):
	# print("iPhon11 Wide SR by using Tele)
	wide_dir = args.wide_dir
	tele_dir = args.tele_dir

	filenames = os.listdir(tele_dir)

	# ----------------------------------------------------------------------------------------------
	save_path_tele2wide = args.Tele_savePath
	if not os.path.exists(save_path_tele2wide):
		os.makedirs(save_path_tele2wide)

	save_path_wideCrop = args.WideCrop_savePath
	if not os.path.exists(save_path_wideCrop):
		os.makedirs(save_path_wideCrop)

	save_path_wideCropBic = args.WideCrop_savePath + '_bic'
	if not os.path.exists(save_path_wideCropBic):
		os.makedirs(save_path_wideCropBic)
	# ----------------------------------------------------------------------------------------------

	for filename in filenames:
		wide_img = os.path.join(wide_dir, filename)
		tele_img = os.path.join(tele_dir, filename)
		sift_align_wide2tele_iphone11(tele_img, wide_img, filename, save_path_tele2wide, save_path_wideCrop, save_path_wideCropBic)


if __name__ == '__main__':

	# -----------------------------------------------------------------
	import argparse
	parser = argparse.ArgumentParser(description='SIFTAlignment')
	parser.add_argument('--mode', type=str, default='iphone11_wideSRTele', help='Wide image path')
	parser.add_argument('--wide_dir', type=str, default=r'./WideView', help='Wide image path')
	parser.add_argument('--tele_dir', type=str, default=r'./TeleView', help='Tele image path')
	parser.add_argument('--Tele_savePath', type=str, default=r'./TeleView_SIFTAlign', help='save path')
	parser.add_argument('--WideCrop_savePath', type=str, default=r'./WideView_crop', help='save path')
	args = parser.parse_args()
	# -----------------------------------------------------------------
	if "iphone11" in args.mode:
		align_iphone11_real_wide2tele(args)
	elif "iphone12_wideSRTele" in args.mode:
		align_iphone12_real_wide2tele(args)
	elif "iphone12_utralSRwide" in args.mode:
		align_iphone12_real_ultra2wide()
	elif "iphone12_utralSRtele" in args.mode:
		align_iphone12_real_utlra2tele()
	else:
		print("---Error Mode!---")



