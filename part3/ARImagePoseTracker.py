import cv2
import numpy as np
import CalibrationHelpers as calib
import glob
import open3d as o3d

def ProjectPoints(points3d, new_intrinsics, R, T):
    points3d = np.hstack((points3d, np.ones((4,1)))).T
    T =  np.reshape(T.T, (3,1))
    RT = np.hstack((R,T))
    points2d = np.matmul(np.matmul(new_intrinsics, RT), points3d)
    return np.array([np.array([points2d[0,0], points2d[1,0]]) * points2d[2,0],
                    np.array([points2d[0,1], points2d[1,1]]) * points2d[2,1],
                    np.array([points2d[0,2], points2d[1,2]]) * points2d[2,2],
                    np.array([points2d[0,3], points2d[1,3]]) * points2d[2,3]]) * 10

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

def FilterByEpipolarConstraint(intrinsics, matches, points1, points2, Rx1, Tx1, threshold = 0.01):
    inlier_mask = []
    E = np.cross(Tx1, Rx1, axisa =0, axisb = 0)
    cx = intrinsics[0][2]
    fx = intrinsics[0][0]
    cy = intrinsics[1][2]
    fy = intrinsics[1][1]
    image1_points = np.float32([points1[m.queryIdx].pt \
                                  for m in matches])

    image2_points = np.float32([points2[m.trainIdx].pt \
                                  for m in matches])

    for uv1, uv2 in zip(image1_points, image2_points):
        x1 = np.array([(uv1[0]-cx)/fx,(uv1[1]-cy)/fy, 1])
        x2 = np.array([(uv1[0]-cx)/fx,(uv1[1]-cy)/fy, 1])
        if np.abs(np.matmul(x2.T, np.matmul(E, x1))) > threshold:
            inlier_mask.append(0)
        else:
            inlier_mask.append(1)

    return inlier_mask

def PointCloudHelper(intrinsics, matches, points1, points2, Rx1, Tx1, mask):
    cx = intrinsics[0][2]
    fx = intrinsics[0][0]
    cy = intrinsics[1][2]
    fy = intrinsics[1][1]
    matches = [matches[i] for i in range(len(mask)) if mask[i] == 1]
    image1_points = np.float32([points1[m.queryIdx].pt \
                                  for m in matches])
    image2_points = np.float32([points2[m.trainIdx].pt \
                                  for m in matches])
    N = len(matches)
    M = np.zeros((N, 3*(N+1)))

    for idx,UV in enumerate(zip(image1_points, image2_points)):
        uv1, uv2 = UV
        x1 = np.array([(uv1[0]-cx)/fx,(uv1[1]-cy)/fy, 1])
        x2 = np.array([(uv1[0]-cx)/fx,(uv1[1]-cy)/fy, 1])
        xT = np.cross(x2, Tx1, axisa = 0, axisb = 0)
        M[idx][N*3] = xT[0]
        M[idx][N*3+1] = xT[1]
        M[idx][N*3+2] = xT[2]
        xRx = np.matmul(np.cross(x2, Rx1, axisa = 0, axisb = 0), x1)
        M[idx][idx*3] = xRx[0]
        M[idx][idx*3+1] = xRx[1]
        M[idx][idx*3+2] = xRx[2]

    W,U,Vt = cv2.SVDecomp(M)
    depths = Vt[-1,:]/Vt[-1,-1]
    your_pointCloud = []
    for idx, uv1 in enumerate(image1_points):
        x1 = np.array([(uv1[0]-cx)/fx,(uv1[1]-cy)/fy, 1])
        X =  depths[idx] * x1
        your_pointCloud.append(X)
    your_pointCloud =  np.asarray(your_pointCloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(your_pointCloud)
    o3d.visualization.draw_geometries([pcd])




##############################################################################
#
#
# Script
#
#
#
##############################################################################
directory = 'images'
images = glob.glob(directory+'/*.JPG')

# Load the reference image that we will try to detect in the webcam
reference = cv2.imread('ARTrackerImage.jpg',0)
RES = 480
reference = cv2.resize(reference,(RES,RES))
feature_detector = cv2.BRISK_create(octaves=5)
reference_keypoints, reference_descriptors = \
        feature_detector.detectAndCompute(reference, None)
keypoint_visualization = cv2.drawKeypoints(
        reference,reference_keypoints,outImage=np.array([]),
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Keypoints",keypoint_visualization)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
intrinsics, distortion, new_intrinsics, roi = \
        calib.LoadCalibrationData('calibration_data')

# Load image 1
fname = images[0]
print(fname)
img = cv2.imread(fname)
img = cv2.resize(img, None,fx=0.25,fy=0.25)
current_frame = cv2.undistort(img, intrinsics, distortion, None,\
                              new_intrinsics)
x, y, w, h = roi
current_frame = current_frame[y:y+h, x:x+w]
current_keypoints, current_descriptors = feature_detector.detectAndCompute(current_frame, None)
#filter out features that we don't care about:
feature_track = {}
for fname in images[1:]:
    new_img = cv2.imread(fname)
    new_img = cv2.resize(new_img, None,fx=0.25,fy=0.25)
    new_frame = cv2.undistort(new_img, intrinsics, distortion, None,\
                                  new_intrinsics)
    x, y, w, h = roi
    new_frame = new_frame[y:y+h, x:x+w]
    new_keypoints, new_descriptors = feature_detector.detectAndCompute(new_frame, None)
    matches = matcher.match(current_descriptors, new_descriptors)
    for m in matches:
        if m.queryIdx in feature_track:
            feature_track[m.queryIdx] = feature_track[m.queryIdx] + 1
        else:
            feature_track[m.queryIdx] = 1
filtered_keypoints = []
filtered_descriptors = []
for key, val in feature_track.items():
    if val >= 2:
        filtered_keypoints.append(current_keypoints[key])
        filtered_descriptors.append(current_descriptors[key])
filtered_descriptors = np.asarray(filtered_descriptors)
filtered_keypoints = np.asarray(filtered_keypoints)
#plot compared to reference image
matches = matcher.match(reference_descriptors, filtered_descriptors)
referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                              for m in matches])
SCALE = 0.1
referencePoints = SCALE*referencePoints/RES
imagePoints = np.float32([filtered_keypoints[m.trainIdx].pt \
                              for m in matches])

ret, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
                                      imagePoints)
R1r = R
T1r = T
image1 = current_frame
image1_keypoints = filtered_keypoints
image1_descriptors = filtered_descriptors
Rx1 = R
Tx1 = T
match_visualization = cv2.drawMatches(reference, reference_keypoints,
                                    current_frame,filtered_keypoints,
                                    matches, 0,
                                    flags= cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('matches',match_visualization)
k = cv2.waitKey(1)

#load all other images
for fname in images[1:]:
    print(fname)
    img = cv2.imread(fname)
    img = cv2.resize(img, None,fx=0.25,fy=0.25)
    current_frame = cv2.undistort(img, intrinsics, distortion, None,\
                                  new_intrinsics)
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]

    current_keypoints, current_descriptors = feature_detector.detectAndCompute(current_frame, None)

    if(len(current_keypoints)>0):
        matches = matcher.match(reference_descriptors, current_descriptors)
        referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                      for m in matches])
        SCALE = 0.1
        referencePoints = SCALE*referencePoints/RES
        imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                      for m in matches])
        ret, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
                                              imagePoints)
        if(ret):
            render_frame = renderCube(current_frame,new_intrinsics,R,T)
            Rx1 = np.matmul(R, R1r.T)
            Tx1 = T - np.matmul(np.matmul(R, R1r.T), T1r)
            matches = matcher.match(image1_descriptors, current_descriptors)
            inlier_mask= FilterByEpipolarConstraint(new_intrinsics, matches, image1_keypoints, current_keypoints, Rx1, Tx1,
                                           threshold = 0.003)

            match_visualization = cv2.drawMatches(
                image1, image1_keypoints,
                current_frame,
                current_keypoints, matches, 0,
                matchesMask =inlier_mask, #this applies your inlier filter
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            cv2.imshow('matches',match_visualization)
            k = cv2.waitKey(1)
            if k == 27 or k==113:
                break
            PointCloudHelper(new_intrinsics, matches, image1_keypoints, current_keypoints, Rx1, Tx1, inlier_mask)

        else:
            print("could not render!")
