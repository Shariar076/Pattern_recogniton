import numpy as np
import cv2

img = np.zeros((512, 512, 3), np.uint8)

cv2.rectangle(img, (0, 0), (50, 50), (255, 0, 0), 5)

cv2.imshow('f',img)
cv2.waitKey(0)