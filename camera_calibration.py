import numpy as np
import cv2 as cv
import glob
import os

chessboardSize = (9,9)
frameSize = (3840,2160)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

objp = np.zeros((chessboardSize[0]*chessboardSize[1],3),np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

objPoints = []
imgPoints = []

images = glob.glob('frames/*.jpg')

for image in  images:
    print(image)
    img = cv.imread(image)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray,chessboardSize, None)

    if ret == True:
        print('detected')
        objPoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)

        width = int(img.shape[1] * 0.25)
        height = int(img.shape[0] * 0.25)
        dim = (width, height)

        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

        cv.imshow('Scaled Down Image', resized)
        filename = os.path.basename(image)
        save_path = 'output/' + filename
        cv.imwrite(save_path, img)
        cv.waitKey(1000)

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


img = cv.imread('frames/chessboard_4.jpg')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('results/calibresult_4_1.png', dst)

# undistort
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('results/calibresult_4_2.png', dst)

mean_error = 0
for i in range(len(objPoints)):
    imgpoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
 
print( "total error: {}".format(mean_error/len(objPoints)) )