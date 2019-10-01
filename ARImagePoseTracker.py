import cv2
import numpy as np
import CalibrationHelpers as calib


# This function is yours to complete
# it should take in a set of 3d points and the intrinsic matrix
# rotation matrix(R) and translation vector(T) of a camera
# it should return the 2d projection of the 3d points onto the camera defined
# by the input parameters
def ProjectPoints(points3d, new_intrinsics, R, T):
    points3d = np.hstack((points3d, np.ones((4,1)))).T
    T =  np.reshape(T.T, (3,1))
    RT = np.hstack((R,T))
    points2d = np.matmul(np.matmul(new_intrinsics, RT), points3d)
    return np.array([np.array([points2d[0,0], points2d[1,0]]) * points2d[2,0],
                    np.array([points2d[0,1], points2d[1,1]]) * points2d[2,1],
                    np.array([points2d[0,2], points2d[1,2]]) * points2d[2,2],
                    np.array([points2d[0,3], points2d[1,3]]) * points2d[2,3]]) * 10

# This function will render a cube on an image whose camera is defined
# by the input intrinsics matrix, rotation matrix(R), and translation vector(T)
def renderCube(img_in, new_intrinsics, R, T):
    # Setup output image
    img = np.copy(img_in)

    # We can define a 20cm cube by 4 sets of 3d points
    # these points are in the reference coordinate frame
    scale = 0.1
    face1 = np.array([[0,0,0],[0,0,scale],[0,scale,scale],[0,scale,0]],
                     np.float32)
    face2 = np.array([[0,0,0],[0,scale,0],[scale,scale,0],[scale,0,0]],
                     np.float32)
    face3 = np.array([[0,0,scale],[0,scale,scale],[scale,scale,scale],
                      [scale,0,scale]],np.float32)
    face4 = np.array([[scale,0,0],[scale,0,scale],[scale,scale,scale],
                      [scale,scale,0]],np.float32)
    # using the function you write above we will get the 2d projected
    # position of these points
    face1_proj = ProjectPoints(face1, new_intrinsics, R, T)
    # this function simply draws a line connecting the 4 points
    img = cv2.polylines(img, [np.int32(face1_proj)], True,
                              tuple([255,0,0]), 3, cv2.LINE_AA)
    # repeat for the remaining faces
    face2_proj = ProjectPoints(face2, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face2_proj)], True,
                              tuple([0,255,0]), 3, cv2.LINE_AA)

    face3_proj = ProjectPoints(face3, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face3_proj)], True,
                              tuple([0,0,255]), 3, cv2.LINE_AA)

    face4_proj = ProjectPoints(face4, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face4_proj)], True,
                              tuple([125,125,0]), 3, cv2.LINE_AA)
    return img

# This function takes in an intrinsics matrix, and two sets of 2d points
# if a pose can be computed it returns true along with a rotation and
# translation between the sets of points.
# returns false if a good pose estimate cannot be found
def ComputePoseFromHomography(new_intrinsics, referencePoints, imagePoints):
    # compute homography using RANSAC, this allows us to compute
    # the homography even when some matches are incorrect
    homography, mask = cv2.findHomography(referencePoints, imagePoints,
                                          cv2.RANSAC, 5.0)
    # check that enough matches are correct for a reasonable estimate
    # correct matches are typically called inliers
    MIN_INLIERS = 30
    if(sum(mask)>MIN_INLIERS):
        # given that we have a good estimate
        # decompose the homography into Rotation and translation
        # you are not required to know how to do this for this class
        # but if you are interested please refer to:
        # https://docs.opencv.org/master/d9/dab/tutorial_homography.html
        RT = np.matmul(np.linalg.inv(new_intrinsics), homography)
        norm = np.sqrt(np.linalg.norm(RT[:,0])*np.linalg.norm(RT[:,1]))
        RT = -1*RT/norm
        c1 = RT[:,0]
        c2 = RT[:,1]
        c3 = np.cross(c1,c2)
        T = RT[:,2]
        R = np.vstack((c1,c2,c3)).T
        W,U,Vt = cv2.SVDecomp(R)
        R = np.matmul(U,Vt)
        return True, R, T
    # return false if we could not compute a good estimate
    return False, None, None

# Load the reference image that we will try to detect in the webcam
reference = cv2.imread('ARTrackerImage.jpg',0)
RES = 480
reference = cv2.resize(reference,(RES,RES))

# create the feature detector. This will be used to find and describe locations
# in the image that we can reliably detect in multiple images
feature_detector = cv2.BRISK_create(octaves=5)
# compute the features in the reference image
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)
# make image to visualize keypoints
# keypoint_visualization = cv2.drawKeypoints(
#         reference,reference_keypoints,outImage=np.array([]))

keypoint_visualization = cv2.drawKeypoints(
        reference,reference_keypoints,outImage=np.array([]),
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# display the image
cv2.imshow("Keypoints",keypoint_visualization)
# wait for user to press a key before proceeding
# cv2.waitKey(0)

# create the matcher that is used to compare feature similarity
# Brisk descriptors are binary descriptors (a vector of zeros and 1s)
# Thus hamming distance is a good measure of similarity
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load the camera calibration matrix
intrinsics, distortion, new_intrinsics, roi = \
        calib.LoadCalibrationData('calibration_data')

# initialize video capture
# the 0 value should default to the webcam, but you may need to change this
# for your camera, especially if you are using a camera besides the default
cap = cv2.VideoCapture(0)

while True:
    # read the current frame from the webcam
    ret, current_frame = cap.read()

    # ensure the image is valid
    if not ret:
        print("Unable to capture video")
        break

    # undistort the current frame using the loaded calibration
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None,\
                                  new_intrinsics)
    # apply region of interest cropping
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]

    # detect features in the current image
    current_keypoints, current_descriptors = \
        feature_detector.detectAndCompute(current_frame, None)
    if(len(current_keypoints)>0):
        # match the features from the reference image to the current image
        matches = matcher.match(reference_descriptors, current_descriptors)
        # matches returns a vector where for each element there is a
        # query index matched with a train index. I know these terms don't really
        # make sense in this context, all you need to know is that for us the
        # query will refer to a feature in the reference image and train will
        # refer to a feature in the current image

        # create a visualization of the matches between the reference and the
        # current image
        match_visualization = cv2.drawMatches(reference, reference_keypoints, current_frame,
                                current_keypoints, matches, 0,
                                flags=
                                cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('matches',match_visualization)
        k = cv2.waitKey(1)
        if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
            #exit
            break


        # set up reference points and image points
        # here we get the 2d position of all features in the reference image
        referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                      for m in matches])
        # convert positions from pixels to meters
        SCALE = 0.1 # this is the scale of our reference image: 0.1m x 0.1m
        referencePoints = SCALE*referencePoints/RES

        imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                      for m in matches])
        # compute homography
        ret, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
                                              imagePoints)
        render_frame = current_frame

        if(ret):
            # compute the projection and render the cube
            render_frame = renderCube(current_frame,new_intrinsics,R,T)

        # display the current image frame
        cv2.imshow('frame', render_frame)
        k = cv2.waitKey(1)
        if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
            #exit
            break
