from ximea import xiapi
import cv2
import numpy as np
import time
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
data = np.load('parametre_kamery.npz')

mtx = data['mtx']
dist = data['dist']
#"""


c = 0

VZD_K_OBJEKTU_CM = 30.0 #dolezite
fx = mtx[0, 0]
fy = mtx[1, 1]

while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image, (616, 514))
    image = cv2.undistort(image, mtx, dist, None, mtx)
    cimg = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=120, param2=50, minRadius=5, maxRadius=500)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cX, cY, radius = i[0], i[1], i[2]

            #Bonus
            priemer_px = 2 * radius
            priemer_cm = (priemer_px * VZD_K_OBJEKTU_CM) / fx

            cv2.circle(cimg, (cX, cY), radius, (0, 255, 0), 4)
            cv2.circle(cimg, (cX, cY), 5, (0, 0, 255), -1)

            text = f"Kruh D:{priemer_cm:.1f}cm"
            cv2.putText(cimg, text, (cX - 40, cY - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    edged = cv2.Canny(blurred, 10, 30)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        #b2
        x, y, w_px, h_px = cv2.boundingRect(approx)
        sirka_cm = (w_px * VZD_K_OBJEKTU_CM) / fx
        vyska_cm = (h_px * VZD_K_OBJEKTU_CM) / fy

        if len(approx) == 3:
            cv2.drawContours(cimg, [approx], 0, (255, 0, 0), 4)
            cv2.circle(cimg, (cX, cY), 5, (0, 0, 255), -1)
            text = f"Trojuholnik {sirka_cm:.1f}x{vyska_cm:.1f}cm"
            cv2.putText(cimg, text, (cX - 60, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        elif len(approx) == 4:
            aspect_ratio = float(w_px) / h_px
            if 0.8 <= aspect_ratio <= 1.2:
                label = "Stvorec"
                color = (0, 0, 255)
            else:
                label = "Obdlznik"
                color = (255, 255, 0)

            cv2.drawContours(cimg, [approx], 0, color, 4)
            cv2.circle(cimg, (cX, cY), 5, (0, 0, 255), -1)
            text = f"{label} {sirka_cm:.1f}x{vyska_cm:.1f}cm"
            cv2.putText(cimg, text, (cX - 60, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Grayscale", gray)
    cv2.imshow("Canny", edged)
    cv2.imshow("Original", image)
    cv2.imshow("Detekcia Tvarov", cimg)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        file_name = f"img_{c}.jpg"
        cv2.imwrite(file_name, image)
        print(f"Uložené: {file_name}")
        c += 1
    time.sleep(0.1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        file_name = f"img_{c}.jpg"
        cv2.imwrite(file_name, image)
        print(f"Uložené: {file_name}")
        c += 1
    time.sleep(0.1)

cv2.destroyAllWindows()

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

print('Done.')