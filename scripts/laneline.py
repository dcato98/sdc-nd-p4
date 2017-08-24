import numpy as np

class Laneline():
    """Line class for tracking history of detected lines"""
    
    def __init__(self):
        # useful constants
        self.image_shape = None
        self.fps = None
        # line detection history
        self.detected = False
        self.detection_misses = 0
        self.left_detection_history = []
        self.right_detection_history = []
        # left/right and world/pixel-space fitted lane line histories, fit: coefficients of 2nd order polynomial
        self.left_fit_pixel_history = []
        self.right_fit_pixel_history = []
        self.left_fit_world_history = []
        self.right_fit_world_history = []
        # radius of curvature history
        self.left_radii_history = []
        self.right_radii_history = []
        # offset history, offset: distance (in meters) of vehicle center from the center of the lane
        self.offset_history = []
        # x values of the last 10 fits of the line
        self.left_x_px_rolling_avg_history = []
        self.right_x_px_rolling_avg_history = []
        
    def record_fit(self, x_pixel, y_pixel, lane, xm_per_pixel, ym_per_pixel):
        """Records new lane line detection measurements."""
        # fit x(y) in pixel-space
        new_fit_pixel = np.polyfit(y_pixel, x_pixel, 2)
        fit_history = self._which_fits(lane, 'pixel')
        fit_history.append(new_fit_pixel)
        
        # fit x(y) in world-space
        new_fit_world = np.polyfit(y_pixel*ym_per_pixel, x_pixel*xm_per_pixel, 2)
        fit_history = self._which_fits(lane, 'world')
        fit_history.append(new_fit_world)
            
        # calculate average of last 10 fits
        y_points = np.array(range(0, self.image_shape[0]))
        new_rolling_avg = self.eval_smooth_fit(y_points, lane, 'pixel', last_n_fits=10)
        rolling_avg_history = self._which_rolling_avgs(lane)
        rolling_avg_history.append(new_rolling_avg)
        
        # calculate new radius of curvature
        y_eval = self.image_shape[0] - 1
        new_radius = self.eval_radius_of_curvature(y_eval, lane, 'world')
        radii_history = self._which_radii(lane)
        radii_history.append(new_radius)
        
        # Determine whether line was detected or not based on radius of curvature differing by more than the allowed tolerance
        if len(radii_history) > 1:
            n_radii = np.min((5, len(radii_history)))
            weights = np.exp(np.divide(range(n_radii),2)) # ~ [1, 1.5, 2.5, 4, 7]
            last_n_radii = radii_history[len(radii_history)-n_radii:]
            wt_avg_last_five_radii = np.average(last_n_radii, weights=weights)
            tolerance_factor = 2
            # max(new_radius, wt_avg_last_five_radii) < 2000 and...
            #radius_problem = ((new_radius / wt_avg_last_five_radii > tolerance_factor) 
            #                  or (wt_avg_last_five_radii / new_radius > tolerance_factor))
            radius_problem = 1.5 < np.abs(np.log(new_radius) - np.log(wt_avg_last_five_radii))
            if radius_problem:
                self.line_miss(lane)
                print("frame i:", len(radii_history)-1, "MISS: radius differs by too much, wt avg radius:", wt_avg_last_five_radii, "new radius:", new_radius)
            else:
                self.line_hit(lane)
        else:
            self.line_hit(lane)
        
        both_lanes_detected = (len(self.left_fit_pixel_history) == len(self.right_fit_world_history))
        if both_lanes_detected:
            # calculate offset from center
            new_offset = self.eval_offset(xm_per_pixel)
            self.offset_history.append(new_offset)
        
        return
    
    def record_no_fit(self):
        """Record that insufficient pixels were found to fit."""
        # skip if we haven't ever detected lane lines before
        if len(self.left_fit_pixel_history) == 0:
            return
        
        # Record that the lines were undetected
        self.line_miss('left')
        self.line_miss('right')
        
        # left/right and world/pixel-space fitted lane line histories, fit: coefficients of 2nd order polynomial
        self.left_fit_pixel_history.append(self.left_fit_pixel_history[-1])
        self.right_fit_pixel_history.append(self.right_fit_pixel_history[-1])
        self.left_fit_world_history.append(self.left_fit_world_history[-1])
        self.right_fit_world_history.append(self.right_fit_world_history[-1])
        # radius of curvature history
        self.left_radii_history.append(self.left_radii_history[-1])
        self.right_radii_history.append(self.right_radii_history[-1])
        # offset history, offset: distance (in meters) of vehicle center from the center of the lane
        self.offset_history.append(self.offset_history[-1])
        # x values of the latest n fits of the line
        self.left_x_px_rolling_avg_history.append(self.left_x_px_rolling_avg_history[-1])
        self.right_x_px_rolling_avg_history.append(self.right_x_px_rolling_avg_history[-1])
    
    def line_miss(self, lane):
        """Records that a detection was missed."""
        self.detection_misses += 1
        if self.detection_misses > 3:
            self.detected = False
        detections = self._which_detections(lane)
        detections.append(0)
        
    def line_hit(self, lane):
        """Records that a line was detected."""
        self.detection_misses = 0
        self.detected = True
        detections = self._which_detections(lane)
        detections.append(1)
        
    def eval_offset(self, xm_per_pixel):
        """Returns the offset of the center of the car from the center of the lane line."""
        y_eval = self.image_shape[0] - 1
        left_x_car = self.eval_fit(y_eval, 'left', 'pixel')
        right_x_car = self.eval_fit(y_eval, 'right', 'pixel')
        center_lane_x = (left_x_car + right_x_car) / 2
        center_car_x = self.image_shape[1] / 2
        offset = (center_lane_x - center_car_x) * xm_per_pixel
        return offset
    
    def eval_fit(self, y_eval, lane, space):
        """Returns the x-value(s) for the given lane line and coordinate space at the specified y-value(s)."""
        fit_history = self._which_fits(lane, space)
        fit = fit_history[-1]
        x = self.eval_polynomial(fit, y_eval)
        return x
        
    def eval_smooth_fit(self, y_points, lane, space, last_n_fits=5):
        """Returns the average x-value(s) of the last `n_fits` for the given lane line and coordinate space at the specified y-value(s)."""
        fits = self._which_fits(lane, space)
        detections = self._which_detections(lane)
        
        # Only use the last `n_fits` fits
        if len(fits) > last_n_fits:
            fits = fits[len(fits)-last_n_fits:]
        
        # Find average x_fit
        x_points = np.zeros_like(y_points).astype(np.float64)
        y_squared = y_points**2
        count = 0
        for i, fit in enumerate(fits):
            x_points += self.eval_polynomial(fit, y_points)
            count += 1
        x_points /= count
        
        return x_points.astype(np.uint8)
    
    def eval_radius_of_curvature(self, y_eval, lane, space):
        """Returns the radius of curvature for the given lane line and coordinate space at the specified y-value."""
        fits = self._which_fits(lane, space)
        fit = fits[-1]
        
        radius = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
        return radius
    
    def eval_polynomial(self, coefficients, x):
        """
        Evaluates an n-degree polynomial y(x) at the specified x-value. 
        
        Parameters:
            `x` - either a single value or a numpy array
            `coefficients` - list of polynomial coefficients (e.g. [A, B, C] for the polynomial Ax^2 + Bx + C)
        """
        if hasattr(x, "__iter__"):
            y = np.zeros_like(x).astype(np.float64)
        else:
            y = 0.0
        for i, coefficient in enumerate(reversed(coefficients.tolist())):
            y += (x ** i) * coefficient
        return y
    
    def _which_radii(self, lane):
        if lane == 'left':
            radii = self.left_radii_history
        elif lane == 'right':
            radii = self.right_radii_history
        else:
            raise ValueError("`lane` must be 'left' or 'right'")
        return radii
    
    def _which_rolling_avgs(self, lane):
        if lane == 'left':
            roll_avgs = self.left_x_px_rolling_avg_history
        elif lane == 'right':
            roll_avgs = self.right_x_px_rolling_avg_history
        else:
            raise ValueError("`lane` must be 'left' or 'right'")
        return roll_avgs
    
    def _which_fits(self, lane, space):
        if lane == 'left':
            fits = (self.left_fit_pixel_history, self.left_fit_world_history)
        elif lane == 'right':
            fits = (self.right_fit_pixel_history, self.right_fit_world_history)
        else:
            raise ValueError("`lane` must be 'left' or 'right'")
        if space == 'pixel':
            fits = fits[0]
        elif space == 'world':
            fits = fits[1]
        else:
            raise ValueError("`space` must be 'pixel' or 'world'")
        return fits
            
    def _which_detections(self, lane):
        if lane == 'left':
            detections = self.left_detection_history
        elif lane == 'right':
            detections = self.right_detection_history
        else:
            raise ValueError("`lane` must be 'pixel' or 'world'")
        return detections