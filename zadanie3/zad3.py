import cv2
import numpy as np
import matplotlib.pyplot as plt


def custom_canny(image, low_threshold=0.05, high_threshold=0.15):
    """
    Vlastná implementácia Cannyho detektora hrán.
    """
    # 1. Gaussovo filtrovanie (redukcia šumu)
    # Manuálne vytvorené jadro (kernel) 5x5 s hodnotou sigma=1.4
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

    # 4. Non-maximum suppression (Potlačenie nemaximálnych hodnôt)
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180  # Uhly len v intervale [0, 180]

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            # Uhol 0 stupňov (horizontálna hrana)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = G[i, j + 1]
                r = G[i, j - 1]
            # Uhol 45 stupňov (diagonála)
            elif (22.5 <= angle[i, j] < 67.5):
                q = G[i + 1, j - 1]
                r = G[i - 1, j + 1]
            # Uhol 90 stupňov (vertikálna hrana)
            elif (67.5 <= angle[i, j] < 112.5):
                q = G[i + 1, j]
                r = G[i - 1, j]
            # Uhol 135 stupňov (opačná diagonála)
            elif (112.5 <= angle[i, j] < 157.5):
                q = G[i - 1, j - 1]
                r = G[i + 1, j + 1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0

    # 5. Dvojité prahovanie (Double thresholding)
    high_thresh = Z.max() * high_threshold
    low_thresh = high_thresh * low_threshold

    res = np.zeros((M, N), dtype=np.int32)
    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= high_thresh)
    weak_i, weak_j = np.where((Z <= high_thresh) & (Z >= low_thresh))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # 6. Hysterézia (Sledovanie hrán)
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if res[i, j] == weak:
                # Ak má aspoň jedného "silného" suseda, stane sa z neho silný pixel
                if ((res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or (res[i + 1, j + 1] == strong)
                        or (res[i, j - 1] == strong) or (res[i, j + 1] == strong)
                        or (res[i - 1, j - 1] == strong) or (res[i - 1, j] == strong) or (res[i - 1, j + 1] == strong)):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    return np.uint8(res)


# --- HLAVNÝ PROGRAM ---

# 1. Načítanie testovacieho obrázka (.jpg / .bmp)
# Odporúčam pre účely testovania obrázok bez zložitého pozadia.
image_path = 'test_image.jpg'  # Zmeňte na vašu cestu k obrázku
img_color = cv2.imread(image_path)

if img_color is None:
    print("Obrázok sa nenašiel! Skontrolujte cestu.")
else:
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 2. Aplikovanie vlastnej implementácie
    custom_edges = custom_canny(img_gray, low_threshold=0.05, high_threshold=0.15)

    # 3. Aplikovanie ekvivalentného algoritmu pomocou OpenCV
    # OpenCV Canny vyžaduje absolútne hodnoty prahov (napr. 50, 150),
    # takže ich nastavíme tak, aby zhruba zodpovedali našim pomerom z max. hodnoty.
    opencv_edges = cv2.Canny(img_gray, 50, 150)

    # 4. Vizuálne porovnanie výsledkov
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Originál (Grayscale)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(custom_edges, cmap='gray')
    plt.title('Vlastný Canny detektor')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(opencv_edges, cmap='gray')
    plt.title('OpenCV Canny')
    plt.axis('off')

    plt.tight_layout()
    plt.show()