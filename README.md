# CarND-AdvancedLaneLines-P4

![alt text][image0]
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./media/output_project_video.gif "Output gif"
[image1]: ./media/Calibration_Image.png "Original vs Undistorted"
[image2]: ./media/Corrected_RGB_Image.png "Road Transformed"
[image3]: ./media/HLS_Sat_Mag_Sobel_Binary.png "Binary Examples"
[image4]: ./media/Final_Binary_Warped_Binary.png "Warp Example"
[image5]: ./media/Overlay_Before_After.png "Output"
[video1]: ./project_video.mp4 "Video"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. The follow is the required writeup.

Here I'll explain in detail how the project was completed and my thought process along the way.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code used to compute the camera matrix and distortion coefficients is contained in the file 'create_cal_pickle.py'
When the script is run the directory containing the calibration images is passed as an argument. The script will create a numpy array containing all of the internal corner point coordinates in a 6x9 chessboard image (objp). As it steps through each image in the directory it will run the 'cv2.findChessboardCorners' function. If the corners are found both 'objp' and the 'corners' list are appended to the master lists of 'objpoints' and 'imgpoints' respectively. Once the image list has been exhausted the 'objpoints' and 'imgpoints' are passed to the 'cv2.calibrateCamera' function to generate the camera matrix and the distortion coefficients. These are then added to a pickle file and saved in the working directory to be loaded later. 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The main working file is 'video_pipeline_oop.py'. This is where they main pipeline of the project lives. Before running the pipeline it first opens the pickle file referenced above and loads the camera matrix and distortion coefficients. It then creates a 'Frame' object and initializes them as attributes. The first method on line 19 generates an undistorted image frame from the input frame using the camera matrix and distortion coefficients. It does this with the 'cv2.undistort' function. Below is an example.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the 'Frame' class there multiple methods to create binaries. From lines 29-69 binaries of the following types are created: grayscale threshold, HSL saturation threshold, red channel threshold, magnitude of image gradient and angle of image gradient. Each is generated in it's own method. Once all of the binaries are generated, they are combined into one with yet other threshold to determine which pixels to keep. Minimum of 2 pixels was found to work well. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In the 'Frame' class there is a method called 'warp_binary(self)' which is used to warp the final binary up to a "top down view." This is located in the 'video_pipeline_oop.py' file at line 108. In the method, the final binary is passed through a masking function to strip out any pixels way from the lane area. 'cv2.getPerspectiveTransform' is then used with the source and destination points (src, dst) to generate the transformation matrix (M). M is then used with the 'cv2.warpPerspective' function to transform the binary to a "top down view."


The source and destination points were hardcoded in the following manner:
```python
src = np.float32([[520, 500],[760, 500],[250, 677],[1030, 677]])
dst = np.float32([[300,400],[980,400],[300,720],[980,720]])
```

The perspective transform was verified to be working as expected by plotting lines using the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the 'Line' class there are three methods used to find lane-line pixels, convert them to points and generate a polynomial. Starting at line 190 there is a 'Line' method called 'histogram_slice' which given it's inputs, slices a binary image up into horizontal sections and searches for a line from the bottom up. If a concentration of pixels are found in a known good window, a corresponding XY point is placed into the master points lists. 

At line 233 is the 'generate_points' method. This method takes the size of the binary image and creates the search window heights and widths to feed to the histogram method. It recieves the output from 'histogram_slice' and if the points are good it makes the final call to append them to the master points lists. Once the master lists are over an initialized limit it will start deleting the first added elements. This gives the line a smoothing effect.

At line 253 is the 'generate_polyline' method. It simplely kicks off the previous two methods, then tries to generate a second order polynomial using the master points lists. If there are not enough points it returns 'None' so an overlay isn't attempted.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

At line 265 the 'Line' class method 'calculate_curve' starts. This was approached very simliar to the course materials. Here is the function in full with comments to explain the process.
```python
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
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final result is done by calling the 'update_frame' method on the object. Here is an example.

![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
