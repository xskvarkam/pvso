from ximea import xiapi
import cv2
import numpy as np
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
#cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print('Opening first camera...')
#cam.open_device()

# settings
#cam.set_exposure(10000)
#cam.set_param("imgdataformat","XI_RGB32")
#cam.set_param("auto_wb",1)

#print('Exposure was set to %i us' %cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisitionq
print('Starting data acquisition...')
#cam.start_acquisition()

c=0;
"""
while cv2.waitKey() != ord('q'):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(240,240))
    cv2.imshow("test", image)
    cv2.imwrite("img" + str(c) + ".jpg", image)
    cv2.waitKey()


    if(c > 2):
        break;
    c += 1
#"""
imgs=[];
for i in range(4):
    imgs.append(cv2.imread("img"+str(i)+".jpg"))
    print(i)
imagus= cv2.imread("img0.jpg")

cv2.imshow("zigo", imagus)
kernel1 = np.array([[1/16,1/8,1/16],
                    [1/8,1/4,1/8],
                    [1/16,1/8,1/16]])
#toto je gaussovsky kernel ktory mierne rozmaze obrazkok.


vis = np.concatenate((imgs[0], imgs[1]), axis=1)

vis2 = np.concatenate((imgs[2], imgs[3]), axis=1)
vis3 = np.concatenate((vis, vis2), axis=0)


vis3[0:240 , 0:240 ] = cv2.filter2D(src=vis3[0:240 , 0:240], ddepth=-1, kernel=kernel1,borderType=cv2.BORDER_DEFAULT)
#gaussovsky


for y in range(240):
    for x in range(y,240):
        temp = vis3[0:240 , 240:480][y][x].copy()

        vis3[0:240 , 240:480][y][x] = vis3[0:240 , 240:480][x][y]
        vis3[0:240, 240:480][x][y]=temp
    vis3[0:240, 240:480][y] = np.flip(vis3[0:240 , 240:480][y],0)


vis3[240:480, 0:240, 0] = 0
vis3[240:480, 0:240, 1] = 0
print(f"Dátový typ: {vis3.dtype}")
print(f"Rozmer: {vis3.shape}")
print(f"Veľkosť: {vis3.size}")
cv2.imshow("Mozaika",vis3)
cv2.waitKey()
cv2.imwrite("mozaika.jpg", vis3)


#cv2.imshow("zigo",vis3)
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