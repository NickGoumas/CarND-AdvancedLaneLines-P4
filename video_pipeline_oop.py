import os
import numpy as np
import argparse
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import *

class Frame(object):

	def __init__(self, mtx, dist, src, dst):
		# Initialize camera cal and warp attributes.
		self.mtx = mtx
		self.dist = dist
		self.src = src
		self.dst = dst

	def undistorted_RGB(self):
		# Undistort the RGB_frame to create a corrected image.
		undst_RGB = cv2.undistort(self.RGB_frame, self.mtx, self.dist, None, self.mtx)
		return undst_RGB

	def undistorted_grayscale(self):
		# Converted the corrected image to grayscale.
		gray = cv2.cvtColor(self.undst_RGB, cv2.COLOR_RGB2GRAY)
		return gray

	def gray_binary(self, min_thresh=200, max_thresh=255):
		# Generate a binary image from grayscale.
		gray_binary = np.zeros_like(self.gray)
		gray_binary[(self.gray > min_thresh) & (self.gray <= max_thresh)] = 1
		return gray_binary

	def saturation_binary(self, min_thresh=150, max_thresh=255):
		# Convert the corrected RGB_image to HLS, then create a binary
		# using only the 'S' layer.
		undst_HLS = cv2.cvtColor(self.undst_RGB, cv2.COLOR_RGB2HLS)
		s_image = undst_HLS[:,:,2]
		s_binary = np.zeros_like(s_image)
		s_binary[(s_image > min_thresh) & (s_image <= max_thresh)] = 1
		return s_binary

	def red_binary(self, min_thresh=225, max_thresh=255):
		# Create a binary image using only the 'R' layer 
		# of the corrected RGB image.
		r_image = self.undst_RGB[:,:,0]
		r_binary = np.zeros_like(r_image)
		r_binary[(r_image > min_thresh) & (r_image <= max_thresh)] = 1
		return r_binary

	def mag_sobel_binary(self, min_thresh=30, max_thresh=130, sobel_kernel=3):
		# Create a binary image using the magnitude of the iamge gradients.
		sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		abs_sobelxy = np.sqrt( (sobelx ** 2) + (sobely ** 2) )
		scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
		mag_binary = np.zeros_like(scaled_sobel)
		mag_binary[(scaled_sobel >= min_thresh) & (scaled_sobel <= max_thresh)] = 1
		return mag_binary

	def ang_sobel_binary(self, sobel_kernel=9, thresh=(0.7,1.4)):
		# Create a binary image using the image gradient angles.
		sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
		ang_binary = np.zeros_like(grad_dir)
		ang_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
		return ang_binary

	def combine_binaries(self, min_pixels=2):
		# Stack all the binaries on top of each other. Each pixel 
		# stack is polled, if it is >= min_pixels the resulting 
		# binary pixel is lit.
		combined_binary = np.zeros_like(self.gray)
		binary_stack = (self.gray_binary()
					+ self.red_binary()
					+ self.saturation_binary()
					+ self.mag_sobel_binary()
					+ self.ang_sobel_binary())
		combined_binary[(binary_stack >= min_pixels)] = 1
		return combined_binary

	def crop_lane_area(self):
		# Create a mask to only focus accept input roughly from 
		# lane line areas.
		binary = self.combine_binaries()
		imshape = binary.shape
		horiz_buffer = 80
		peak_height = 380
		vertices = np.array([[(0+horiz_buffer,imshape[0]),
			                 ((imshape[1]/2),peak_height),
		    	     (imshape[1]-horiz_buffer,imshape[0]),
		        	       ((imshape[1]*.30), imshape[0]),
	                	            ((imshape[1]/2), 550),
	                 	 ((imshape[1]*.70), imshape[0])]],
		dtype=np.int32)
		mask = np.zeros_like(binary)
		if len(binary.shape) > 2:
			channel_count = binary.shape[2]
			ignore_mask_color = (255,) * channel_count
		else:
			ignore_mask_color = 255
		cv2.fillPoly(mask, vertices, ignore_mask_color)
		masked_image = cv2.bitwise_and(binary, mask)
		return masked_image

	def warp_binary(self):
		# Warp the image to create a 'top-down' view.
		binary = self.crop_lane_area()
		img_size = (binary.shape[1], binary.shape[0])
		M = cv2.getPerspectiveTransform(self.src, self.dst)
		warped_binary = cv2.warpPerspective(binary, M, img_size, flags=cv2.INTER_LINEAR)
		return warped_binary

	def overlay_lane(self):
		# Initialize points list, img_size and a zero array.
		points = []
		img_size = (self.RGB_frame.shape[1], self.RGB_frame.shape[0])
		overlay_mask = np.zeros_like(self.RGB_frame)

		# If the left and right line objects exist, proceed with 
		# generating points and creating a polyfill to overlay 
		# on the zero array as if it was a 'top-down' view.
		if leftLine.polyline and rightLine.polyline:
			for y in np.linspace(0, self.RGB_frame.shape[0]):
				points.append([int(leftLine.polyline(y)), int(y)])
			for y in np.linspace(self.RGB_frame.shape[0], 0):
				points.append([int(rightLine.polyline(y)), int(y)])
			cv2.fillPoly(overlay_mask, np.array([points], dtype=np.int32), (0,255,0))

		# Create the inverse perspective matrix using dst and src
		# and warp the zero array with it's polyfill back down to 
		# the same perspective as the original RGB image. Then stack
		# the images on top of each other. Polyfill is transparent.
		M = cv2.getPerspectiveTransform(self.dst, self.src)
		overlay_mask = cv2.warpPerspective(overlay_mask, M, img_size, flags=cv2.INTER_LINEAR)
		result = cv2.addWeighted(self.RGB_frame, 1, overlay_mask, 0.3, 0)
		return result

	def overlay_text(self, img):
		# if left and right line objects exist calculate the curve
		# of each and the midpoint between the base of each curve. 
		# The difference between the midpoint and image center is 
		# the offset.
		if leftLine.polyline and rightLine.polyline:
			LeftCurveTxt = 'Left Curve:{}m'.format(leftLine.calculate_curve())
			RightCurveTxt = 'Right Curve:{}m'.format(rightLine.calculate_curve())
			xm_per_pix = 3.7/700.0
			xcenter = float(leftLine.line_bottom_point() + rightLine.line_bottom_point())/2.0
			offset = ((1280/2.0) - xcenter) * xm_per_pix
			OffsetTxt = 'Offset from Center:{}m'.format(round(offset, 2))

			# Use putText to overlay the text on the img passed to the function.
			cv2.putText(img, text=LeftCurveTxt, org=(20,30), 
				fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0),thickness=2)
			cv2.putText(img, text=RightCurveTxt, org=(20,60),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0),thickness=2)
			cv2.putText(img, text=OffsetTxt, org=(20,90),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0),thickness=2)

	def update_frame(self, new_RGB_frame):
		# For each new frame passed, update all the needed images.
		self.RGB_frame = new_RGB_frame
		self.undst_RGB = self.undistorted_RGB()
		self.gray = self.undistorted_grayscale()
		self.warped_binary = self.warp_binary()
		# Generate new line equations.
		leftLine.generate_polyline(self.warped_binary)
		rightLine.generate_polyline(self.warped_binary)
		# Highlight the lane and place the text.
		self.lane_overlay = self.overlay_lane()
		self.overlay_text(self.lane_overlay)
		return self.lane_overlay

class Line(object):

	def __init__(self, side):
		# Initialize the 'side' of the line as well as the lists
		# of X and Y points used to fit the polynomial eq. The 
		# list_length is the number of point pairs used in the 
		# running average of the polynomial. This smooths the
		# movement of the line.
		self.side = side
		self.Xlist = []
		self.Ylist = []
		self.list_length = 150
		self.polyline = None

	def histogram_slice(self, binary_image, vert_center, height, window_width):
		# This method accepts a binary image to detect a line in. The 
		# vert_center is the vertical center of the window the histogram
		# will be generated from. The window top and bottom are derived 
		# from this. The height is the window height. Window width is
		# window_width as well.
		top = max(vert_center - height // 2, 0)
		bottom = min(vert_center + height // 2, binary_image.shape[0])

		# If we are searching for a point in the image along the bottom
		# row OR we have not found a point yet, ignore window_width and
		# search half of the whole image.
		if bottom == binary_image.shape[0] or len(self.Xlist) < 1:
			histogram = np.sum(binary_image[top:bottom,:], axis=0)
			midpoint = np.int(histogram.shape[0]/2)
			if self.side == 'left' and np.var(histogram[:midpoint]) > 0:
				Xval = np.argmax(histogram[:midpoint])
				#print('argmax:', Xval)
				return [Xval,vert_center]
			elif self.side == 'right' and np.var(histogram[midpoint:]) > 0:
				Xval = np.argmax(histogram[midpoint:]) + midpoint
				return [Xval,vert_center]
			else:
				return [None, None]

		# Once we're away from the bottom of the image we can start
		# trying to follow the line by only looking in the window_width
		# range.
		elif bottom != binary_image.shape[0]:
			histogram = np.sum(binary_image[top:bottom,:], axis=0)
			midpoint = np.int(histogram.shape[0]/2)
			left, right = self.Xlist[-1]-window_width//2, self.Xlist[-1]+window_width//2
			if self.side == 'left' and np.var(histogram[left:right]) > 0:
				Xval = np.argmax(histogram[left:right]) - window_width//2 + self.Xlist[-1]
				#print('argmax:', Xval)
				return [Xval,vert_center]
			elif self.side == 'right' and np.var(histogram[left:right]) > 0:
				Xval = np.argmax(histogram[left:right]) - window_width//2 + self.Xlist[-1]
				return [Xval,vert_center]
			else:
				return [None, None]


	def generate_points(self, binary_image):
		# Pass a list of values to the histogram_slice method to determine
		# how many slices to take and how wide the vertical slices are to be.
		# We start at the bottom and work up because the line is usually
		# more obvious in the bottom of the image.
		for i in range(binary_image.shape[0], 0, -binary_image.shape[0]//10):

			# Return the X,Y pair for a given slice. If they exist, append 
			# them to the master list. If the master list is over it's 
			# delete the oldest (first) element.
			pair = self.histogram_slice(binary_image, i, binary_image.shape[0]//10, 100)
			if pair[0]:
				self.Xlist.append(pair[0])
				self.Ylist.append(pair[1])
				#print('List:', pair[0])
			if len(self.Xlist) > self.list_length:
				del self.Xlist[0]
				del self.Ylist[0]


	def generate_polyline(self, binary_image):
		# If the master X,Y list has more than two points, generate
		# a polynomial function and return it. If not, return None.
		self.generate_points(binary_image)
		if len(self.Xlist) > 1:
			xpoints = np.array(self.Xlist)
			ypoints = np.array(self.Ylist)
			self.polyline = np.poly1d(np.polyfit(ypoints, xpoints, 2))
		else:
			self.polyline = None
		return self.polyline

	def calculate_curve(self):
		# Rough conversions given in the project to map
		# pixels to meters in the X and Y directions.
		ym_per_pix = 30.0/720.0
		xm_per_pix = 3.7/700.0
		xpoints = np.array(self.Xlist)*xm_per_pix
		ypoints = np.array(self.Ylist)*ym_per_pix
		# Generate polynomial coefficient container
		poly_function = np.poly1d(np.polyfit((ypoints),(xpoints), 2))
		# Set coefficients.
		A, B, C = poly_function.c
		y = 720 # Most interested in curve at the bottom of the image.
		# Equation for the curve radius from poly coe. 
		curve = (1.0+((2.0*A*y+B)**2.0)**3.0/2.0)/abs(2.0*A)
		return round(curve, 0)

	def line_bottom_point(self):
		# Calculate the X value of the line where it crosses the 
		# bottom of the image. This is used to find the car center
		# later on.
		point = self.polyline(720)
		return int(point)

# Grab video filename.
parser = argparse.ArgumentParser(description='Get arg filename.')
parser.add_argument('file')
args = parser.parse_args()
arg_filename = args.file

# Load camera calibration values from pickle file.
calibration_pickle = pickle.load(open('calibration_pickle.p', 'rb'))
mtx = calibration_pickle['mtx']
dist = calibration_pickle['dist']

# Set source and destination points for perspective warping.
# TopLeft, TopRight, BottomLeft, BottomRight. Hand picked points.
src = np.float32([[520, 500],[760, 500],[250, 677],[1030, 677]])
dst = np.float32([[300,400],[980,400],[300,720],[980,720]])

# Create frame object and two line objects.
frame = Frame(mtx, dist, src, dst)
leftLine = Line('left')
rightLine = Line('right')


# If the file is a jpg show the concatenated image of two images 
# from the pipeline. Two images must have equal channels.
if args.file.split('.')[-1] == 'jpg':
	frame.update_frame(mpimg.imread(args.file))
	plt.imshow( np.concatenate((frame.gray_binary(200,255), frame.red_binary(225,255) ), axis=1), )
	#plt.title('Final Binary and Warped Final Binary', fontsize=18)
	plt.axis('off')
	plt.show()

# Else if the file is an mp4, run the overlay frame
# by frame and output a new video.
elif args.file.split('.')[-1] == 'mp4':
	input_clip = VideoFileClip(arg_filename)
	output_filename = '2output_' + arg_filename
	# Only read first three seconds for testing.
	#input_clip = input_clip.subclip(0,1)
	new_clip = input_clip.fl_image(frame.update_frame)
	new_clip.write_videofile(output_filename, audio=False)

else:
	print('File type not known...')