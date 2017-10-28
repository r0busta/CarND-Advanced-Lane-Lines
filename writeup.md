# Advanced Lane Finding

## Writeup

*Advanced Lane Finding Project*

The goals/steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, to create a threshold binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/undistort_output.png "Undistorted"
[image2]: ./images/undistorted_example.png "Undistorted"
[image3]: ./images/binary_combo.png "Binary Example"
[image4]: ./images/transformed.png "Road Transformed"
[image5]: ./images/color_fit_lines.png "Fit Visual"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup/README

*Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.*

You're reading it!

### Camera Calibration

*Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.*

The code for this step is contained in the `calibrate()` function of the file called `proccess.py` (further, all code mentions refer to that file).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted][image1]

### Pipeline (single images)

*1. Provide an example of a distortion-corrected image.*

After camera calibration, the processing pipeline can apply the distortion correction to the processed images like this one:
![Undistorted][image2]

*2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.*

I used a combination of HLS channel thresholds to generate a binary image (`get_binary()` function).  
Here's an example of my output for this step.

![Binary example][image3]

*3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.*

The code for my perspective transform includes functions `transform()`, `get_transform_matr()` and `get_inv_transform_matr()`.
The function `get_transform_matr()` calculates transformation matrix from source (`TRANSFORM_SRC`) and destination (`TRANSFORM_DST`) points.
I chose to hardcode the source and destination points. 
I verified that my perspective transformation function `transform()` was working as expected by drawing the `TRANSFORM_SRC` and `TRANSFORM_DST` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective transformation][image4]

*4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?*

To find lane-lines pixels, I chose the sliding window method and histogram peaks finding. 
In the function `detect_lane()`, for a given binary warped image `binary_warped`, I step through 9 windows to find histogram peaks, i. e. lane lines.
After finding all potential line-pixel, I calculate a 2nd order polynomial that fits the pixels.   

*5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.*

The radius of curvature of the line and the position of the vehicle calculated in `get_curvature_m()` and `get_offset_m()` functions. 
The function `get_curvature_m` calculates radius (in meters) of curvature of left and right lines.
The function `get_offset_m()` calculates offset of the car from the lane center axis.
The result is used by `detect_lane()` to print both calculated values on the output image.

*6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.*

I implemented this step in `draw_lane()` function.
Here is an example of my result on a test image:

![Final result][image5]

---

### Pipeline (video)

*Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).*

Here's a [link to my video result](./project_video.mp4) (or at [YouTube](https://youtu.be/2buaf-LOxtM))

---

### Discussion

*Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?*

The weakest point of the pipeline is the step where lane lines are filtered out from the rest of the image, i. e. creation of thresholded binary image.
As one can see in the [project video](./project_video.mp4), even in a relatively good conditions, the pipeline shows flickering lane drawing when light conditions changes a bit.

To make the pipeline more robust, more sophisticated image pre-processing can be added, e. g. using Sobel operator for calculating binary layers.  

  
