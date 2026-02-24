from ximea import xiapi
import cv2
import numpy as np
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
#cam.open_device_by_SN('41305651')
# (open by serial number)
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
#"""
while 0 and cv2.waitKey() != ord('q'):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(257,308))
    cv2.imshow("Foto", image)
    cv2.imwrite("img" + str(c) + ".jpg", image)


    if(c > 12):
        break;
    c += 1
import numpy as np
import cv2 as cv
import glob
import os
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (5, 7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (5, 7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# --- SPLNENIE ZADANIA ---
if ret:
    # 1. Vypíšte maticu vnútorných parametrov kamery
    print("\n--- MATICA VNÚTORNÝCH PARAMETROV KAMERY ---")
    print(mtx)

    # 2. Určte hodnoty fx, fy, cx, cy
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print("\n--- VNÚTORNÉ PARAMETRE ---")
    print(f"fx (Ohnisková vzdialenosť x): {fx:.4f}")
    print(f"fy (Ohnisková vzdialenosť y): {fy:.4f}")
    print(f"cx (Optický stred x): {cx:.4f}")
    print(f"cy (Optický stred y): {cy:.4f}")

    print(f"\nDistorzné koeficienty:\n{dist}")

    # 3. Uložte maticu kamery a distorzné koeficienty pre ďalšie použitie
    np.savez("parametre_kamery.npz", mtx=mtx, dist=dist)
    print("\nMatica a koeficienty boli úspešne uložené do 'parametre_kamery.npz'.")

    # 4. Demonštrujte odstránenie skreslenia (undistortion) na reálnom obraze
    print("\n--- DEMONŠTRÁCIA UNDISTORTION ---")
    # Načítame prvý obrázok zo zoznamu ako testovací
    test_img = cv.imread(images[0])

    if test_img is not None:
        # Aplikovanie opravenia obrazu
        undistorted_img = cv.undistort(test_img, mtx, dist, None, mtx)

        # Zobrazenie vedľa seba (ak sú obrázky príliš veľké, môžeš pridať cv.resize)
        cv.imshow('Originalny obraz (so skreslenim)', test_img)
        cv.imshow('Opraveny obraz (Undistorted)', undistorted_img)

        print("Stlač ľubovoľnú klávesu v okne s obrázkom pre ukončenie...")
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Chyba: Kalibrácia zlyhala. Nenašiel sa dostatok dobrých bodov.")
cv.destroyAllWindows()


"""
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
for i in range(10):
     #get data and pass them from camera to img
    cam.get_image(img)
    image = img.get_image_data_numpy()
    cv2.imshow("test", image)
    cv2.waitKey()
#     #get raw data from camera
#     #for Python2.x function returns string
#     #for Python3.x function returns bytes
    data_raw = img.get_image_data_raw()
#
#     #transform data to list
    data = list(data_raw)
#
#     #print image data and metadata
#     print('Image number: ' + str(i))
#     print('Image width (pixels):  ' + str(img.width))
#     print('Image height (pixels): ' + str(img.height))
#     print('First 10 pixels: ' + str(data[:10]))
#     print('\n')
"""
# stop data acquisition
print('Stopping acquisition...')
#cam.stop_acquisition()

# stop communication
#cam.close_device()

print('Done.')