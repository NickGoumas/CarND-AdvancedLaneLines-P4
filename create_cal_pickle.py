import os
import numpy as np
import argparse
import cv2
import pickle
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Get cal images path.')
parser.add_argument('path')
args = parser.parse_args()


cal_image_list = os.listdir(args.path)
#print(type(os.getcwd()))
#print(cal_image_list)

def calibrate_camera(cal_image_list):
	# Prepare object points, (0,0,0), (1,0,0), (2,0,0)...
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

	# Arrays to store object and image points.
	objpoints = []
	imgpoints = []

	# Iterate though the list and look for chessboard corners.
	for fname in cal_image_list:
		fname = os.getcwd() + '/' + args.path + '/' + fname

		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find corners.
		ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

		# When corners are found append objp and corners
		if ret == True:
			print('Corners found:', fname)
			objpoints.append(objp)
			imgpoints.append(corners)

			cv2.drawChessboardCorners(img, (9,6), corners, ret)
			#cv2.imshow('img', img)
			#cv2.waitKey(500)

	# Create flipped image size.
	img_size = (img.shape[1], img.shape[0])
	# Calibrate camera with object and image points.
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

	# Save calibration values as a pickle file.
	calibration_pickle = {}
	calibration_pickle['mtx'] = mtx
	calibration_pickle['dist'] = dist
	calibration_pickle['rvecs'] = rvecs
	calibration_pickle['tvecs'] = tvecs
	pickle.dump(calibration_pickle, open('calibration_pickle.p', 'wb'))

calibrate_camera(cal_image_list)