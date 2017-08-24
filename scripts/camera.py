import cv2
import dill as pickle
import numpy as np
import os

class Camera():
    """
    Camera class for calibration and simple functions.
    
    Sample Usage:
        # create camera
        camera = Camera(name='my_nexus_6P', directory='/mnt/c/Nexus6P')
        
        # calibrate camera
        n_vert, n_horiz = 6, 9 # number of inner chessboard corners in the vertical and horizontal directions, respectively
        camera.calibrate(calibration_folder='chessboard_images', n_vert, n_horiz)
        
        # undistort an image from a calibrated camera 
        undistorted = camera.undistort(image)
        
        # save camera to file
        camera.save()
        
        # automatically restore a saved camera by creating new instance with same name and path
        del camera
        camera = Camera(name='my_nexus_6P', directory='/mnt/c/Nexus6P')
        
        # alternatively, manually restore a saved camera from file
        camera.load()
    """
    def __init__(self, name, directory):
        # resolve camera filename
        self.name = name.replace(' ', '_')
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self.directory = directory
        self.filename = os.path.join(self.directory, self.name + '.pkl')
        
        # restore camera from file, if possible
        if os.path.exists(self.filename):
            self._load()
        # otherwise, initialize calibration constants
        else:
            self.mtx = None
            self.dist = None
            self.isCalibrated = False
    
    def calibrate(self, calibration_image_folder, n_vert, n_horiz, file_suffix='.jpg'):
        """Calculates the camera's calibration coefficients using chessboard images."""
        calibration_image_path = os.path.join(self.directory, calibration_image_folder)
        image_paths = [os.path.join(calibration_image_path, filename) for filename in os.listdir(calibration_image_path) 
                       if filename.endswith(file_suffix)]
        calibration_coefficients = self._calibrate_camera_with_chessboard(image_paths, n_vert, n_horiz)
        if calibration_coefficients:
            self.mtx = calibration_coefficients[0]
            self.dist = calibration_coefficients[1]
            self.isCalibrated = True
    
    def undistort_image(self, image):
        """Undistorts an image using the camera's calibration coefficients."""
        if self.isCalibrated:
            undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        else:
            undistorted = image.copy()
        return undistorted
    
    def undistort_image_generator(self, images):
        """Returns a generator which yields undistorted images"""
        for image in images:
            yield self.undistort_image(image)

    def save(self):
        """Saves the camera to a pickled file."""
        with open(self.filename, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)
    
    def _load(self):
        """Loads the camera from a pickled file."""
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                tmp_dict = pickle.load(f)
            self.__dict__.update(tmp_dict)
    
    def _calibrate_camera_with_chessboard(self, calibration_image_paths, nx, ny):
        """
        Calculates camera calibration parameters from a list of checkerboard images.

        Parameters:
            img_filenames - list of chessboard image filenames
            nx = number of inner chessboard corners in the x direction
            ny = number of inner chessboard corners in the y direction

        Returns:
            mtx - camera matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            dist - distortion coefficients [k1, k2, p1, p2, k3]
            rvecs - ???
            tvecs - ???
        """
        # prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(ny,nx,0)
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane
        
        # Step through the list and search for chessboard corners
        for idx, filename in enumerate(calibration_image_paths):
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            return (mtx, dist, rvecs, tvecs)
        return None
