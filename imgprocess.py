import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance

img1 = cv2.imread('E:\\CCISR\\VesselDRIVE\\training\\original\\images\\29.png')
img2 = cv2.imread('E:\\CCISR\\VesselDRIVE\\training\\image\\29.jpg')

res = cv2.addWeighted(img1, 0.55, img2, 0.8, 0.9)
plt.imshow(res)
plt.show()