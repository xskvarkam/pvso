import cv2
import numpy as np


def custom_canny(image, low_threshold=0.05, high_threshold=0.15):
    """
    Vlastná implementácia Cannyho detektora hrán.
    """
    # 1. Gaussovo filtrovanie
    kernel = np.array([[2, 4, 5, 4, 2],
                       [4, 9, 12, 9, 4],
                       [5, 12, 15, 12, 5],
                       [4, 9, 12, 9, 4],
                       [2, 4, 5, 4, 2]], dtype=np.float32) / 159.0
    smoothed = cv2.filter2D(image, -1, kernel)

    # 2. Výpočet gradientu (Manuálny Sobel operátor)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = cv2.filter2D(smoothed, cv2.CV_64F, Kx)
    Iy = cv2.filter2D(smoothed, cv2.CV_64F, Ky)

    # 3. Výpočet veľkosti a smeru gradientu
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255  # Normalizácia
    theta = np.arctan2(Iy, Ix)

    # 4. Non-maximum suppression
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

    # 5. Dvojité prahovanie
    high_thresh = Z.max() * high_threshold
    low_thresh = high_thresh * low_threshold
    res = np.zeros((M, N), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= high_thresh)
    weak_i, weak_j = np.where((Z <= high_thresh) & (Z >= low_thresh))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # 6. Hysterézia
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


# --- HLAVNÝ PROGRAM ---

image_path = 'test_image.jpg'  # Zmeňte na vašu cestu k obrázku
img_color = cv2.imread(image_path)

if img_color is None:
    print("Obrázok sa nenašiel! Skontrolujte cestu.")
else:
    # Konverzia do odtieňov sivej
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Vypočítanie hrán
    custom_edges = custom_canny(img_gray, low_threshold=0.05, high_threshold=0.15)
    opencv_edges = cv2.Canny(img_gray, 50, 150)

    # --- VIZUÁLNE POROVNANIE CEZ OpenCV ---

    # Pre vizualizáciu textu si vytvoríme kópie, aby sme neničili pôvodné dáta
    disp_gray = img_gray.copy()
    disp_custom = custom_edges.copy()
    disp_cv2 = opencv_edges.copy()

    # Nastavenie textu (font, veľkosť, farba - biela 255, hrúbka)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(disp_gray, 'Original', (10, 30), font, 1, 255, 2, cv2.LINE_AA)
    cv2.putText(disp_custom, 'Vlastny Canny', (10, 30), font, 1, 255, 2, cv2.LINE_AA)
    cv2.putText(disp_cv2, 'OpenCV Canny', (10, 30), font, 1, 255, 2, cv2.LINE_AA)

    # Spojenie obrázkov vedľa seba (horizontálne)
    combined_result = np.hstack((disp_gray, disp_custom, disp_cv2))

    # Voliteľné: Ak je originálny obrázok príliš veľký, zmenšíme výsledné okno, aby sa zmestilo na obrazovku
    height, width = combined_result.shape
    max_width = 1600  # Maximálna šírka okna v pixeloch
    if width > max_width:
        scale = max_width / width
        combined_result = cv2.resize(combined_result, (max_width, int(height * scale)))

    # Zobrazenie výsledku
    cv2.imshow('Porovnanie Canny detektorov', combined_result)

    print("Stlacte lubovolnu klavesu v okne s obrazkom pre ukoncenie...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()