from __future__ import print_function

import cv2
import numpy as np
import os

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
		# print('Key points in input image: %d' % len(keypoints1))
		# print('Key points in ref image  : %d' % len(keypoints2))
		# print('Matched points           : %d' % len(matches))
	
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

def sift_align_syn(path_ref, path_lr, name, scale, save_path):
	print(path_ref, path_lr)

	im_ref = cv2.imread(path_ref)
	im_lr = cv2.imread(path_lr)
	# print(im_lr.shape, im_ref.shape)

	lh, lw, c = im_lr.shape
	im_lr_bic = cv2.resize(im_lr,  (int(lw*scale), int(lh*scale)), interpolation=cv2.INTER_CUBIC)

	align_ref = alignImages(im_ref, im_lr_bic, VERBOSE=True)
	print(os.path.join(save_path, name))
	cv2.imwrite(os.path.join(save_path, name), align_ref)

def align_dir(lr_dir, ref_dir, scale, save_path):
	LR_dir = lr_dir
	Ref_dir = ref_dir
	filenames = os.listdir(Ref_dir)
	scale = scale
	print(filenames)

	save_path = save_path
	print(save_path)
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	for filename in filenames:
		lr_img = os.path.join(LR_dir, filename)
		if "JPEG" in LR_dir:
			lr_img = os.path.join(LR_dir, filename[:-4]+'.jpg')

		ref_img = os.path.join(Ref_dir, filename)
		sift_align_syn(ref_img, lr_img, filename, scale, save_path)


if __name__ == '__main__':
	# -----------------------------------------------------------------
	import argparse
	parser = argparse.ArgumentParser(description='SIFTAlignment')
	parser.add_argument('--LR_dir', type=str, default=r'./WideView_iso2x_JPEG75_crop', help='LR wide image path')
	parser.add_argument('--Ref_dir', type=str, default=r'./TeleView_crop', help='Crop Tele image save path')
	parser.add_argument('--save_path', type=str, default=r'./WideView_iso2x_JPEG75_crop_SIFTAlign', help='save path')
	parser.add_argument('--scale', type=int, default=2, help='downsample scale')
	args = parser.parse_args()
	# -----------------------------------------------------------------
	LR_dir = args.LR_dir
	Ref_dir = args.Ref_dir
	save_path = args.save_path
	scale = args.scale

	align_dir(LR_dir, Ref_dir, scale, save_path)





