# Imports
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
        
def calibrate_camera_with_chessboard(img_filenames, nx, ny, draw=False):
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
    for idx, filename in enumerate(img_filenames):
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            if draw:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        return (mtx, dist, rvecs, tvecs)
    return None

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Calculate the x or y pixel-wise gradient within a threshold.
    """
    # convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    
    # scale to 8-bit [0, 255]
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Calculate the magnitude of the pixel-wise gradient within a threshold.
    """
    # convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # calculate gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    
    # scale to 8-bit [0, 255]
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    
    # Apply threshold
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Calculate a direction of the pixel-wise gradient within a threshold.
    """
    # convert to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Calculate gradient direction
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    # Apply threshold
    dir_binary = np.zeros_like(dir_sobel)
    dir_binary[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return dir_binary

def channel_thresh(channel, thresh=(0,255)):
    """
    Parameters:
        channel - a single-channel of an image, e.g. a grayscale image or 'R'-channel of RGB image
        threshold - a tuple containing the threshold bounds, inclusive
    """
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary

def warp_perspective(img, src, dest):
    """
    Undistort and unwarp an image.
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    warped = undist.copy()
    imgx, imgy, imgz = img.shape
    border = 100
    src = np.float32([corners[0,0], corners[nx-1, 0], 
                      corners[-1, 0], corners[-nx, 0]])
    dst = np.float32([[border, border], [imgy-border, border], 
                      [imgy-border, imgx-border], [border, imgx-border]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(warped, M, (imgy, imgx), flags=cv2.INTER_LINEAR)
        
    return warped, M