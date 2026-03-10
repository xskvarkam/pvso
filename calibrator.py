from ximea import xiapi
import cv2
import numpy as np
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
#cam.open_device_by_SN('41305651')
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(10000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisitionq
print('Starting data acquisition...')
cam.start_acquisition()

c=0;

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image, (308 * 2, 257 * 2))

    cv2.imshow("Foto", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        cv2.imwrite(f"img{c}.jpg", image)
        print(f"Uložené: img{c}.jpg")
        c += 1

    elif key == ord('q'):
        break

cv2.destroyAllWindows()


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0:5, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images=[];
for i in range(14):
    images.append("img"+str(i)+".jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5, 7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (5, 7), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
if ret:
    print("matica")
    print(mtx)

    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print("\nparametre")
    print(f"fx (Ohnisková vzdialenosť x): {fx:.4f}")
    print(f"fy (Ohnisková vzdialenosť y): {fy:.4f}")
    print(f"cx (Optický stred x): {cx:.4f}")
    print(f"cy (Optický stred y): {cy:.4f}")

    print(f"\nDistorzné koeficienty:\n{dist}")

    np.savez("parametre_kamery.npz", mtx=mtx, dist=dist)
    print("\ngoog")
    test_img = cv2.imread(images[0])
    if test_img is not None:
        undistorted_img = cv2.undistort(test_img, mtx, dist, None, mtx)

        cv2.imshow('Original', test_img)
        cv2.imshow('Opraveny', undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("bad.")

cv2.destroyAllWindows()

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

print('Done.')