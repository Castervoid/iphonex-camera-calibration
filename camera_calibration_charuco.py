import numpy as np
import cv2 as cv
import os

CHARUCOBOARD_ROWCOUNT = 11
CHARUCOBOARD_COLCOUNT = 11
SQUARE_LENGTH = 0.015
MARKER_LENGTH = 0.011

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_100)
charuco_board = cv.aruco.CharucoBoard(
    (CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

charuco_detector = cv.aruco.CharucoDetector(charuco_board)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

cap = cv.VideoCapture("video_files/chessboard_aruco.MOV")
frame_idx = 0

def variance_of_laplacian(image):
    return cv.Laplacian(image, cv.CV_64F).var()

blur_threshold = 100
motion_threshold = 20
prev_gray = None

objPoints_all = []
imgPoints_all = []
charuco_ids_all = []

while True:
    ret, img = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 4 != 0:
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if variance_of_laplacian(gray) < blur_threshold:
        continue

    if prev_gray is not None:
        diff = cv.absdiff(gray, prev_gray)
        mean_diff = np.mean(diff)
        if mean_diff > motion_threshold:
            prev_gray = gray
            continue

    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_ids is not None and len(charuco_ids) == 100:
        objPoints_all.append(charuco_board.getChessboardCorners()[charuco_ids.flatten()])
        imgPoints_all.append(charuco_corners)
        charuco_ids_all.append(charuco_ids)

        cv.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
        cv.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

        os.makedirs('output_aruco', exist_ok=True)
        filename = f"frame_{frame_idx:04d}.jpg"
        save_path = 'output_aruco/' + filename
        cv.imwrite(save_path, img)

    prev_gray = gray

cap.release()
cv.destroyAllWindows()

if len(objPoints_all) == 0:
    print("No Charuco corners detected, calibration failed.")
    exit()

ret, cameraMatrix, dist, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(
    charucoCorners=imgPoints_all,
    charucoIds=charuco_ids_all,
    board=charuco_board,
    imageSize=gray.shape[::-1],
    cameraMatrix=None,
    distCoeffs=None,
    flags=cv.CALIB_RATIONAL_MODEL,
    criteria=criteria
)

print("Camera Calibrated: ", ret)
print("\nCamera Matrix:\n", cameraMatrix)
print("\nDistortion Coefficients:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)

np.savez('calibration_data_charuco.npz',
         ret=ret,
         cameraMatrix=cameraMatrix,
         distCoeffs=dist,
         rvecs=np.array(rvecs),
         tvecs=np.array(tvecs))

fs = cv.FileStorage('calibration_charuco.yaml', cv.FILE_STORAGE_WRITE)
fs.write('ret', float(ret))
fs.write('cameraMatrix', cameraMatrix)
fs.write('distCoeffs', dist)
fs.startWriteStruct('rvecs', cv.FILE_NODE_SEQ)
for rvec in rvecs:
    fs.write('', rvec)
fs.endWriteStruct()
fs.startWriteStruct('tvecs', cv.FILE_NODE_SEQ)
for tvec in tvecs:
    fs.write('', tvec)
fs.endWriteStruct()
fs.release()

mean_error = 0
for i in range(len(objPoints_all)):
    imgpoints2, _ = cv.projectPoints(objPoints_all[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints_all[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(objPoints_all)))
