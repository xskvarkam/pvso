import cv2
import numpy as np
from ximea import xiapi

def nas_filter2d(image, kernel):
    img_height, img_width = image.shape
    k_height, k_width = kernel.shape

    pad_h = 2
    pad_w = 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    output = np.zeros_like(image, dtype=np.float64)

    for y in range(img_height):
        for x in range(img_width):
            region = padded_image[y: y + k_height, x: x + k_width]
            output[y, x] = np.sum(region * kernel)

    return output
def canny2(image, low_threshold=0.05, high_threshold=0.15):

    #gaus
    kernel = np.array([[2, 4, 5, 4, 2],
                       [4, 9, 12, 9, 4],
                       [5, 12, 15, 12, 5],
                       [4, 9, 12, 9, 4],
                       [2, 4, 5, 4, 2]], dtype=np.float32) / 159.0
    smoothed = nas_filter2d(image, kernel)

    #gradient
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = nas_filter2d(smoothed, Kx)
    Iy = nas_filter2d(smoothed, Ky)

    #velkost a smer gradientu
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = G[i, j + 1]
                r = G[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = G[i + 1, j - 1]
                r = G[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = G[i + 1, j]
                r = G[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = G[i - 1, j - 1]
                r = G[i + 1, j + 1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0

    # prahy
    high_thresh = Z.max() * high_threshold
    low_thresh = high_thresh * low_threshold
    res = np.zeros((M, N), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= high_thresh)
    weak_i, weak_j = np.where((Z <= high_thresh) & (Z >= low_thresh))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    #pripajanie
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if res[i, j] == weak:
                if ((res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or (res[i + 1, j + 1] == strong)
                        or (res[i, j - 1] == strong) or (res[i, j + 1] == strong)
                        or (res[i - 1, j - 1] == strong) or (res[i - 1, j] == strong) or (res[i - 1, j + 1] == strong)):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    return np.uint8(res)

image_path = 'test_image.jpg'
img_color = cv2.imread(image_path)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

custom_edges = canny2(img_gray, low_threshold=0.05, high_threshold=0.15)
opencv_edges = cv2.Canny(img_gray, 50, 150)

disp_gray = img_gray.copy()
disp_custom = custom_edges.copy()
disp_cv2 = opencv_edges.copy()

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(disp_gray, 'Original', (10, 30), font, 1, 255, 2, cv2.LINE_AA)
cv2.putText(disp_custom, 'Nas Canny', (10, 30), font, 1, 255, 2, cv2.LINE_AA)
cv2.putText(disp_cv2, 'OpenCV Canny', (10, 30), font, 1, 255, 2, cv2.LINE_AA)

combined_result = np.hstack((disp_gray, disp_custom, disp_cv2))

height, width = combined_result.shape
max_width = 1600
if width > max_width:
    scale = max_width / width
    combined_result = cv2.resize(combined_result, (max_width, int(height * scale)))

cv2.imshow('Porovnanie Canny detektorov', combined_result)

cam = xiapi.Camera()

cam.open_device()
cam.set_exposure(10000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

img = xiapi.Image()
cam.start_acquisition()
while True:

    cam.get_image(img)
    frame = img.get_image_data_numpy()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_small = cv2.resize(frame_gray, (240, 240))
    img_small_c = cv2.resize(frame, (240, 240))

    custom_edges = canny2(img_small, low_threshold=0.05, high_threshold=0.15)
    opencv_edges = cv2.Canny(img_small, 50, 150)

    disp_orig = img_small.copy()
    disp_custom = custom_edges.copy()
    disp_cv2 = opencv_edges.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(disp_custom, 'Nas Canny', (10, 20), font, 0.5, 255, 1, cv2.LINE_AA)
    cv2.putText(disp_cv2, 'OpenCV', (10, 20), font, 0.5, 255, 1, cv2.LINE_AA)

    combined_result = np.hstack((disp_custom, disp_cv2))
    combined_result_display = cv2.resize(combined_result,
                                         (combined_result.shape[1] * 2, combined_result.shape[0] * 2))
    cv2.imshow('orig', img_small_c)
    cv2.imshow('Porovnanie', combined_result_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()