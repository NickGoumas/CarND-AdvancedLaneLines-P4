import os
import numpy as np
import argparse
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

parser = argparse.ArgumentParser(description='Get test images path.')
parser.add_argument('path')
args = parser.parse_args()

image_list = os.listdir(args.path)

# Open pickle cal file and extract cal values.
calibration_pickle = pickle.load(open('calibration_pickle.p', 'rb'))
mtx = calibration_pickle['mtx']
dist = calibration_pickle['dist']

# Get full path to first image.
img_fname = fname = os.getcwd() + '/' + args.path + '/' + 'test3.jpg'

def gray_binary(gray, min_thresh, max_thresh):
	binary = np.zeros_like(gray)
	binary[(gray > min_thresh) & (gray <= max_thresh)] = 1
	return binary

def rgb_binary(rgb_image, min_thresh, max_thresh, rgb_letter):
	if rgb_letter == 'r':
		rgb_image = rgb_image[:,:,0]
	elif rgb_letter == 'g':
		rgb_image = rgb_image[:,:,1]
	elif rgb_letter == 'b':
		rgb_image = rgb_image[:,:,2]
	binary = np.zeros_like(rgb_image)
	binary[(rgb_image > min_thresh) & (rgb_image <= max_thresh)] = 1
	return binary

def hls_binary(hls_image, min_thresh, max_thresh, hls_letter):
	if hls_letter == 'h':
		hls_image = hls_image[:,:,0]
	elif hls_letter == 'l':
		hls_image = hls_image[:,:,1]
	elif hls_letter == 's':
		hls_image = hls_image[:,:,2]
	binary = np.zeros_like(hls_image)
	binary[(hls_image > min_thresh) & (hls_image <= max_thresh)] = 1
	return binary

def mag_sobel(gray, min_thresh, max_thresh, sobel_kernel=3):
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	abs_sobelxy = np.sqrt( (sobelx ** 2) + (sobely ** 2) )
	scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
	sobel_binary = np.zeros_like(scaled_sobel)
	sobel_binary[(scaled_sobel >= min_thresh) & (scaled_sobel <= max_thresh)] = 1
	return sobel_binary

def ang_sobel(gray, sobel_kernel=9, thresh=(0.7, 1.4)):
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output = np.zeros_like(grad_dir)
	binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
	return binary_output


# Begin pipeline...
img_RGB = mpimg.imread(img_fname) # Read image as RGB.
undst_RGB = cv2.undistort(img_RGB, mtx, dist, None, mtx)
undst_HLS = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HLS)
gray = cv2.cvtColor(undst_RGB, cv2.COLOR_RGB2GRAY)


f, plots = plt.subplots(4, 3, figsize=(15,15))
plots[0,0].imshow(undst_RGB)
plots[0,0].set_title('Undistorted', fontsize=12)
plots[0,1].imshow(gray, 'gray')
plots[0,1].set_title('Grayscale', fontsize=12)
plots[0,2].imshow(gray_binary(gray, 200, 255), 'gray')
plots[0,2].set_title('Gray Binary', fontsize=12)

plots[1,0].imshow(rgb_binary(undst_RGB,225,255,'r'), 'gray')
plots[1,0].set_title('R Binary', fontsize=12)
plots[1,1].imshow(rgb_binary(undst_RGB,200,255,'g'), 'gray')
plots[1,1].set_title('G Binary', fontsize=12)
plots[1,2].imshow(rgb_binary(undst_RGB,200,255,'b'), 'gray')
plots[1,2].set_title('B Binary', fontsize=12)

plots[2,0].imshow(hls_binary(undst_HLS,150,255,'h'), 'gray')
plots[2,0].set_title('H Binary', fontsize=12)
plots[2,1].imshow(hls_binary(undst_HLS,150,255,'l'), 'gray')
plots[2,1].set_title('L Binary', fontsize=12)
plots[2,2].imshow(hls_binary(undst_HLS,150,255,'s'), 'gray')
plots[2,2].set_title('S Binary', fontsize=12)

plots[3,0].imshow(mag_sobel(gray,30,130,9), 'gray')
plots[3,0].set_title('Mag Sobel, 30-130 k9', fontsize=12)
plots[3,1].imshow(mag_sobel(gray,30,130,11), 'gray')
plots[3,1].set_title('Mag Sobel, 30-130 k11', fontsize=12)
plots[3,2].imshow(ang_sobel(gray), 'gray')
plots[3,2].set_title('Mag Sobel, 30-130 k13', fontsize=12)

#plt.imshow(ang_sobel(gray), 'gray')
plt.show()