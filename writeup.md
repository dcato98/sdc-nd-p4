# Udacity Self-Driving Car Nanodegree
---

## Project #4: Advanced Lane Finding
---

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

[image1]: ./camera/calibration_images/calibration1.jpg "Distorted"
[image2]: ./camera/undistorted/undistorted_img_0.jpg "Undistorted"
[image3]: ./pipeline/output_6_2.png "Thresholds"
[image4]: ./pipeline/output_5_0.png "Perspective Transform"
[image5]: ./pipeline/output_16_1.png "Radius Plot"
[image6]: ./pipeline/output_16_2.png "Offset Plot"
[image7]: ./pipeline/annotated_lane_img_0000000000.jpg "Annotated Lane Line"

[video1]: ./test_videos/project_video_annotated.mp4 "Annotated Project Video"
[video2]: ./test_videos/project_video_composite.mp4 "Composite Project Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Cell 4 in the IPython notebook [pipeline.ipynb](./scripts/pipeline.ipynb) contains the high-level code for camera calibration. A new Camera object is created and, if a pickled Camera object isn't already at the provided path, calibrates it by calling `Camera.calibrate` and by providing a folder of checkerboard images, the number of vertical inner checkerboard points, and the number of horizontal inner checkerboard points.

For a lower level look at what is going on here, see the function `Camera._calibrate_camera_with_chessboard` on line 84 in [camera.py](./scripts/camera.py). Object points are prepared by assuming the chessboard is fixed on the (x, y) plane at z=0 such that the object points are the same for each calibration image. Image points are detected by using `cv2.findChessboardCorners`. finally, `cv2.calibrateCamera()` uses the object points and image points to calculate the camera matrix and distortion coefficients.

Finally, the calibrated Camera object is used to correct distorted images using `Camera.undistort_image`, which is a simple wraper for `cv2.undistort`.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

See the previous point for how distortion correction is done. Here is an example of an image before and after distortion correction:

![distorted][image1]
Before distortion correction

![undistorted][image2]
After distortion correction

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Cell 7 in the IPython notebook [pipeline.ipynb](./scripts/pipeline.ipynb) contains the code where I experimented with a combination of thresholds using thresholding functions from the file [image_utils.py](./scripts/image_utils.py). Once I was satisfied with the combined threshold, I defined a generator in cell 8 for applying this threshold to new images. Here's an example of my output for this step:

![thresholded][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

See cell 5 in [pipeline.ipynb](./scripts/pipeline.ipynb) for the code that calibrates the perspective transform coordinates and computes the transform matrix `M` and inverse transform matrix `Minv`. See cell 6 for the code that validates the transform matrices on curved images.

Using trial and error to find appropriate coordinates, I hardcoded the source and destination points in the following manner:

```python
y, x, _ = image.shape   
src = [[275, 670], 
       [610, 440], 
       [670, 440], 
       [1030, 670]]
dest = [[x//4, y], 
        [x//4, 0], 
        [(x*3)//4, 0], 
        [(x*3)//4, y]]
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 275, 670      | 320, 720      |
| 610, 440      | 320, 0        | 
| 670, 440      | 960, 0        |
| 1030, 670     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Here is an example showing the perspective transform on a image of a curved lane. Notice how the curves appear parallel, indicating that a correct combination of source and destination points were chosen.

![perspective transform][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

Cell 9 in [pipeline.ipynb](./scripts/pipeline.ipynb) defines `find_lane_lines`, a function which uses a sliding window search to identify the position of lane lines in an image. Cell 10 defines `lane_line_generator` which calls `find_lane_lines` to identify the lane location when the previous frame's location is unknown or when the previous frame's radius of curvature differed by a significant amount from the frame before it. When the previous frame's fit is known and within the threshold, `lane_line_generator` simply searches for laneline pixels within 80 pixels of the previous frame's fitted polynomial.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated in `Laneline.eval_radius_of_curvature` in [laneline.py](./scripts/laneline.py) on lines 156 through 162. Since the curves are fitted to a 2nd order polynomial, the radius of curvature is given by the equation,
```python
radius = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```
where `y_eval` is the y-position of the front of the car and `fit` is the list of coefficients in descending polynomial order.

The position of the vehicle with respect to the center of the lane is calculated in `Laneline.eval_offset` in the same file on lines 119 through 127. The offset is calculated as follows,
```python
center_lane_x = (left_x_lane + right_x_lane) / 2
center_car_x = self.image_shape[1] / 2
offset = (center_lane_x - center_car_x) * xm_per_pixel
```
where `left_x_lane` and `right_x_lane` are the x-positions (measured in pixels) of the left and right lane, respectively, and xm_per_pixel is the number of meters per pixel.

Both the radius and the position are calculated for every frame except those where no polynomial fit was possible. In this case, their values are copied forward from the previous frame.

Bonus! Here are plots of the radius of curvature and offset from center measurements over the course of the video:

![radius_plot][image5]

![offset_plot][image6]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The image processing pipeline is defined and executed in cells 11 and 12, respectively, in [pipeline.ipynb](./scripts/pipeline.ipynb). Here is an example of the pipeline output on a test image (note that, in this frame, the average radius-of-curvature is 4652 m and the car offset from center is 0.37 m):

![annotated_laneline][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.

Here's the final project video:

![Annotated Project Video][video1]

Bonus! Here's a composite video simultaneously showing each step of the pipeline:

![Composite Project Video][video2]

---

### Discussion

#### 1. Briefly discuss any issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The first thresholding function I used worked for 95% of the frames, but failed in the two sections of horizontal shadows across the road. I inspected the output of the individual thresholds and noticed that many of these (most noticably, the S-threshold and y-gradient-threshold) were mis-identifying this horizontal section as a lane. To correct this, I applied a directional threshold to the output of each of these, which resolved the problem.

At some point, performance of this pipeline would need to be improved. I profiled the pipeline and noticed that the thresholds take up the majority of time, although my CPU utilization tops out around 40%. Given this, running the thresholds in parallel should provide a significant performance boost.

The pipeline is very likely to fail when another car or other obstacle is blocking the lane lines, or when there are no lane lines. While the use of the S-threshold improves lane detection under varying lighting, it may still fail under night conditions.

One key next step in making the pipeline more robust would be to collect more data in a wider variety of conditions (e.g. rain, snow, night, etc...) and identifying failure conditions to be addressed. Another key next step would be to localize the car's position on a known road map in order to provide higher confidence predictions.
