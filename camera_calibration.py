import numpy as np
import cv2 as cv
import os

chessboardSize = (10,10)
frameSize = (1280,720)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

square_size = 0.015
objp = np.zeros((chessboardSize[0]*chessboardSize[1],3),np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)*square_size

objPoints = []
imgPoints = []

cap = cv.VideoCapture("video_files/chessboard_720p_30fps_11x11_15mm.MOV")
frame_idx = 0

def variance_of_laplacian(image):
    return cv.Laplacian(image, cv.CV_64F).var()

blur_threshold = 150
motion_threshold = 20 

prev_gray = None

while True:
    ret, img = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 5 != 0:
        continue
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    if variance_of_laplacian(gray) < blur_threshold:
        continue  # skip blurry frames
    
    if prev_gray is not None:
        diff = cv.absdiff(gray, prev_gray)
        mean_diff = np.mean(diff)
        if mean_diff > motion_threshold:
            prev_gray = gray
            continue  # skip shaken frames
    
    ret2, corners = cv.findChessboardCorners(gray,chessboardSize, None)
    if ret2:
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners2)
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret2)
        os.makedirs('output', exist_ok=True)
        filename = f"frame_{frame_idx:04d}.jpg"
        save_path = 'output/' + filename
        cv.imwrite(save_path, img)
    
    prev_gray = gray

cap.release()
cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

print("Camera Calibrated: ", ret)
print("\nCamera Matrix:\n", cameraMatrix)
print("\nDistortion Parameters\n",rvecs)
print("\nTranslation Vectors:\n", tvecs)


np.savez('calibration_data.npz',
         ret=ret,
         cameraMatrix=cameraMatrix,
         distCoeffs=dist,
         rvecs=np.array(rvecs),
         tvecs=np.array(tvecs))

fs = cv.FileStorage('calibration.yaml', cv.FILE_STORAGE_WRITE)

fs.write('ret', float(ret))
fs.write('cameraMatrix', cameraMatrix)
fs.write('distCoeffs', dist)

# Write rvecs as a sequence
fs.startWriteStruct('rvecs', cv.FILE_NODE_SEQ)
for rvec in rvecs:
    fs.write('', rvec)
fs.endWriteStruct()

# Write tvecs as a sequence
fs.startWriteStruct('tvecs', cv.FILE_NODE_SEQ)
for tvec in tvecs:
    fs.write('', tvec)
fs.endWriteStruct()

fs.release()


# img = cv.imread('frames/chessboard_720p_30fps_4.jpg')
# h,  w = img.shape[:2]
# newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# undistort
# dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('results/calibresult_4_1.png', dst)

# undistort
# mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imwrite('results/calibresult_4_2.png', dst)

mean_error = 0
for i in range(len(objPoints)):
    imgpoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objPoints)) )

# ... rest of your calibration code unchanged ...
