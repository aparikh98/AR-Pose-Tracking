import cv2
import numpy as np
import glob

# This function records images from the connected camera to specified directory
# when the "Space" key is pressed.
# directory: should be a string corresponding to the name of an existing
# directory
def CaptureImages(directory):
    # Open the camera for capture
    # the 0 value should default to the webcam, but you may need to change this
    # for your camera, especially if you are using a camera besides the default
    cam = cv2.VideoCapture(0)
    img_counter = 0
    # Read until user quits
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        # display the current image
        cv2.imshow("Display", frame)
        # wait for 1ms or key press
        k = cv2.waitKey(1) #k is the key pressed
        if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
            #exit
            break
        elif k == 32: #32 is ascii for space
            #record image
            img_name = "calib_image_{}.png".format(img_counter)
            cv2.imwrite(directory+'/'+img_name, frame)
            print("Writing: {}".format(directory+'/'+img_name))
            img_counter += 1
    cam.release()


# This function calls OpenCV's camera calibration on the directory of images
# created above.
# Returns the following values
# intrinsics: the current camera intrinsic calibration matrix
# distortion: the current distortion coefficients
# roi: the region of the image with full data
# new_intrinsics: the intrinsic calibration matrix of an image after
# undistortion and roi cropping
def CalibrateCamera(directory,visualize=False):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # here we only care about computing the intrinsic parameters of the camera
    # and not the true positions of the checkerboard, so we can do everything
    # up to a scale factor, this means we can prepare our object points as
    # (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) representing the coordinates
    # of the corners in the checkerboard's local coordinate frame
    # if we cared about exact position we would need to scale these according
    # to the true size of the checkerboard
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Set up arrays to store object points and image points from all the images
    # Here the image points are the 2d positions of the corners in each image
    # the object points are the true 3d positions of the corners in the
    # checkerboards coordinate frame
    objpoints = [] # 3d point in the checkerboard's coordinate frame
    imgpoints = [] # 2d points in image plane.
    # Grab all images in the directory
    images = glob.glob(directory+'/*.png')
    for fname in images:
        # read the image
        img = cv2.imread(fname)
        # convert to grayscale (this simplifies corner detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points (after refining them)
        # This ensures that only images where all corners could be detected are
        # used for calibration
        if ret == True:
            # the object points in the checkerboard frame are always the same
            # so we just duplicate them for each image
            objpoints.append(objp)
            # refine corner locations, initial corner detection is performed by
            # sliding a convultional filter across the image, so accuracy is
            # at best 1 pixel, but from the image gradient we can compute the
            # location of the corner at sub-pixel accuracy
            corners2 = \
                cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # append the refined pixel location to our image points array
            imgpoints.append(corners2)
            # if visualization is enabled, draw and display the corners
            if visualize==True:
                cv2.drawChessboardCorners(img, (9,6), corners2, ret)
                cv2.imshow('Display', img)
                cv2.waitKey(500)
    # Perform camera calibration
    # Here I have fixed K3, the highest order distortion coefficient
    # This simplifies camera calibration and makes it easier to get a good
    # result, however this is only works under the assumption that your camera
    # does not have too much distortion, if your camera is very wide field of
    # view, you should remove this flag
    ret, intrinsics, distortion, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, \
                            None,flags=cv2.CALIB_FIX_K3,criteria=criteria)
    # print error if calibration unsuccessful
    if not ret:
        print("Calibration failed, recollect images and try again")
    # if successful, compute an print reprojection error, this is a good metric
    # for how good the calibration is. If your result is greater than 1px you
    # should probably recalibrate
    total_error = 0
    for i in range(len(objpoints)):
        # project the object points onto each camera image and compare
        # against the detected image positions
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], \
                                          intrinsics, distortion)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    print( "mean error: {}".format(total_error/len(objpoints)) )
    # compute the region for where we have full information and the resulting
    # intrinsic calibration matrix
    h,  w = img.shape[:2]
    new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(intrinsics, \
                                                        distortion, (w,h), 1,\
                                                        (w,h))
    # return only the information we will need going forward
    return intrinsics, distortion, roi, new_intrinsics

# This function will save the calibration data to a file in the specified
# directory
def SaveCalibrationData(directory, intrinsics, distortion, new_intrinsics, \
                        roi):
    np.savez(directory+'/calib', intrinsics=intrinsics, distortion=distortion,\
             new_intrinsics = new_intrinsics, roi=roi)

# This function will load the calibration data from a file in the specified
# directory
def LoadCalibrationData(directory):
    npzfile = np.load(directory+'/calib.npz')
    return npzfile['intrinsics'], npzfile['distortion'], \
            npzfile['new_intrinsics'], npzfile['roi']
