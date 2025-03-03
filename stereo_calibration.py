import numpy as np
import cv2 as cv
import glob
import re

def compute_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0
    total_points = 0
    
    for i in range(len(object_points)):
        img_points2, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv.norm(image_points[i], img_points2, cv.NORM_L2) / len(img_points2)
        total_error += error
        total_points += 1
    
    return total_error / total_points

def rectify_and_draw_epilines(img1, img2, points1, points2, F, save_path1, save_path2):
    def draw_lines(img, lines, pts):
        r, c = img.shape[:2]
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for r, pt in zip(lines, pts):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img = cv.line(img, (x0, y0), (x1, y1), color, 1)
            img = cv.circle(img, tuple(map(int, pt.ravel())), 5, color, -1)
        return img
    
    lines1 = cv.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img1_epilines = draw_lines(img1, lines1, points1)
    
    lines2 = cv.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    img2_epilines = draw_lines(img2, lines2, points2)
    
    cv.imwrite(save_path1, img1_epilines)
    cv.imwrite(save_path2, img2_epilines)
    
    return img1_epilines, img2_epilines

# Chessboard parameters
chessboardSize = (8,6)
frameSize = (640,480)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

objpoints, imgpointsL, imgpointsR = [], [], []

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

imagesLeft = sorted(glob.glob('images/stereoLeft/*.png'), key=extract_number)
imagesRight = sorted(glob.glob('images/stereoRight/*.png'), key=extract_number)

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    
    if retL and retR:
        objpoints.append(objp)
        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

if not imgpointsL:
    print("No image points detected! Check image loading and chessboard detection.")
    exit()

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)

reproj_error_left = compute_reprojection_error(objpoints, imgpointsL, rvecsL, tvecsL, cameraMatrixL, distL)
reproj_error_right = compute_reprojection_error(objpoints, imgpointsR, rvecsR, tvecsR, cameraMatrixR, distR)
print(f"Left Camera Reprojection Error: {reproj_error_left:.4f}")
print(f"Right Camera Reprojection Error: {reproj_error_right:.4f}")

flags = cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, cameraMatrixL, distL, cameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

# --- Added print statements for additional camera parameters ---
baseline = np.linalg.norm(trans)
focal_length_left = newCameraMatrixL[0,0]
focal_length_right = newCameraMatrixR[0,0]
fov_x_left = 2 * np.arctan(frameSize[0] / (2 * focal_length_left)) * 180/np.pi
fov_y_left = 2 * np.arctan(frameSize[1] / (2 * newCameraMatrixL[1,1])) * 180/np.pi
fov_x_right = 2 * np.arctan(frameSize[0] / (2 * focal_length_right)) * 180/np.pi
fov_y_right = 2 * np.arctan(frameSize[1] / (2 * newCameraMatrixR[1,1])) * 180/np.pi

print("Stereo Calibration Parameters:")
print(f"Baseline: {baseline:.4f} (same unit as object points)")
print(f"Left Camera Focal Length: {focal_length_left:.4f} pixels")
print(f"Right Camera Focal Length: {focal_length_right:.4f} pixels")
print(f"Left Camera FOV: {fov_x_left:.2f}° (horizontal) x {fov_y_left:.2f}° (vertical)")
print(f"Right Camera FOV: {fov_x_right:.2f}° (horizontal) x {fov_y_right:.2f}° (vertical)")
# --- End of added print statements ---

# Epipolar Constraint Testing
imgL = cv.imread(imagesLeft[0], cv.IMREAD_GRAYSCALE)
imgR = cv.imread(imagesRight[0], cv.IMREAD_GRAYSCALE)
points_left = np.array(imgpointsL[0].reshape(-1, 2))
points_right = np.array(imgpointsR[0].reshape(-1, 2))
img1_lines, img2_lines = rectify_and_draw_epilines(imgL, imgR, points_left, points_right, fundamentalMatrix, "epilines_left.png", "epilines_right.png")

cv.imshow("Left Image Epilines", img1_lines)
cv.imshow("Right Image Epilines", img2_lines)
cv.waitKey(0)

rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap2.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()

cv.destroyAllWindows()
