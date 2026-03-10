from ximea import xiapi
import cv2
import numpy as np

def nic(x):
    pass

cam = xiapi.Camera()
print('Otváram kameru...')
cam.open_device()

cam.set_exposure(10000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

img = xiapi.Image()
print('Spúšťam spracovanie (stlačte "q" pre ukončenie)...')
cam.start_acquisition()

data = np.load('parametre_kamery.npz')
mtx = data['mtx']
dist = data['dist']

cv2.namedWindow('Nastavenia', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Nastavenia', 600, 600)

panel = np.zeros((230, 500, 3), dtype=np.uint8)

cv2.putText(panel, "1 - H Min (Od akej farby)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(panel, "2 - H Max (Do akej farby)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(panel, "3 - S Min (Sytost od bledych)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(panel, "4 - S Max (Sytost do sytych)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(panel, "5 - V Min (Jas od tmy)", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(panel, "6 - V Max (Jas do silneho svetla)", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
cv2.putText(panel, "7 - NOVA FARBA (0=C, 60=Z, 120=M)", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

cv2.imshow('Nastavenia', panel)

cv2.createTrackbar('1', 'Nastavenia', 170, 179, nic)
cv2.createTrackbar('2', 'Nastavenia', 10, 179, nic)
cv2.createTrackbar('3', 'Nastavenia', 100, 255, nic)
cv2.createTrackbar('4', 'Nastavenia', 255, 255, nic)
cv2.createTrackbar('5', 'Nastavenia', 50, 255, nic)
cv2.createTrackbar('6', 'Nastavenia', 255, 255, nic)
cv2.createTrackbar('7', 'Nastavenia', 60, 179, nic)

while True:
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    frame = cv2.resize(frame, (616, 514))

    frame = cv2.undistort(frame, mtx, dist, None, mtx)

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos('1', 'Nastavenia')
    h_max = cv2.getTrackbarPos('2', 'Nastavenia')
    s_min = cv2.getTrackbarPos('3', 'Nastavenia')
    s_max = cv2.getTrackbarPos('4', 'Nastavenia')
    v_min = cv2.getTrackbarPos('5', 'Nastavenia')
    v_max = cv2.getTrackbarPos('6', 'Nastavenia')
    nova_farba_h = cv2.getTrackbarPos('7', 'Nastavenia')

    if h_min <= h_max:
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
    else:
        lower1 = np.array([h_min, s_min, v_min])
        upper1 = np.array([179, s_max, v_max])
        mask1 = cv2.inRange(hsv, lower1, upper1)

        lower2 = np.array([0, s_min, v_min])
        upper2 = np.array([h_max, s_max, v_max])
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    hsv[mask > 0, 0] = nova_farba_h

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('1 - Original', frame)
    cv2.imshow('2 - Maska (Prahovanie)', mask)
    cv2.imshow('3 - Vysledok', result)
    cv2.imshow('Nastavenia', panel)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()
print('Ukončené.')