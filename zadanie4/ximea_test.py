from ximea import xiapi
import cv2

# =============================================================================
# NASTAVENIA - tu môžete meniť parametre kamery
# =============================================================================

EXPOSURE_US      = 10000   # čas expozície v mikrosekundách (napr. 1000 = 1ms)
GAIN_DB          = 0.0     # zosilnenie v dB (0 = žiadne, vyššie = jasnejší obraz)

AUTO_EXPOSURE    = True    # ZAPNUTÉ automatická expozícia a gain
AUTO_WB          = True    # ZAPNUTÉ automatické vyváženie bielej

WB_RED           = 1.0     # Tieto hodnoty si automatika teraz bude riadiť sama
WB_GREEN         = 1.0
WB_BLUE          = 1.0

IMAGE_FORMAT     = "XI_RGB24"   # formát obrazu: "XI_RGB24", "XI_MONO8", "XI_RAW8"

# =============================================================================

cam = xiapi.Camera()
print("Otváranie kamery...")
cam.open_device()

print(f"Kamera: {cam.get_device_name()}  |  SN: {cam.get_device_sn()}")

# Expozícia a gain
if AUTO_EXPOSURE:
    cam.enable_aeag()
    print("Auto expozícia: ZAP")
else:
    cam.disable_aeag()
    cam.set_exposure(EXPOSURE_US)
    cam.set_gain(GAIN_DB)

# Vyváženie bielej
if AUTO_WB:
    cam.enable_auto_wb()
    print("Auto WB: ZAP")
else:
    cam.disable_auto_wb()
    cam.set_wb_kr(WB_RED)
    cam.set_wb_kg(WB_GREEN)
    cam.set_wb_kb(WB_BLUE)

# Formát obrazu
cam.set_imgdataformat(IMAGE_FORMAT)

img = xiapi.Image()
cam.start_acquisition()
print("Snímanie spustené. Stlačte 'q' pre ukončenie.")

while True:
    cam.get_image(img)
    frame = img.get_image_data_numpy(invert_rgb_order=True)

    # ==========================================================
    # OPRAVA FARIEB: Prevod z XIMEA (RGB) do OpenCV (BGR)
    # Týmto predídeme tomu, aby bola modrá a červená prehodená
    # ==========================================================
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (600, 600))
    # Získanie aktuálnych dynamických hodnôt z kamery
    # KROK 2: Získanie aktuálnych dynamických hodnôt z kamery
    akt_exp = cam.get_exposure()
    akt_gain = cam.get_gain()
    akt_r = cam.get_wb_kr()
    akt_g = cam.get_wb_kg()
    akt_b = cam.get_wb_kb()

    # KROK 3: Príprava textov s hodnotami
    text_svetlo = f"Exp: {akt_exp} us  |  Gain: {akt_gain:.2f} dB"
    text_farby = f"R: {akt_r:.2f}  |  G: {akt_g:.2f}  |  B: {akt_b:.2f}"

    # Výpis do konzoly (na jeden riadok)
    print(f"{text_svetlo}  |  {text_farby}      ", end="\r")

    # KROK 4: Vykreslenie parametrov priamo do videa (zeleným písmom do ľavého horného rohu)
    # cv2.putText(obraz, text, (x, y), font, velkost_pisma, farba_bgr(B, G, R), hrubka_ciary)
    cv2.putText(frame, text_svetlo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, text_farby, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # XI_RGB24 je už v skutočnosti BGR, takže OpenCV konverziu nepotrebuje!
    cv2.imshow("XIMEA kamera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("\nUkončujem...") # Nový riadok, aby sme nezmazali posledný stav
cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()
print("Hotovo.")