
#Code to generate binary mode images
import cv2
import numpy as np


img = cv2.imread("stop116.jpg")
cv2.imshow("original",img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
blur = cv2.GaussianBlur(gray,(5,5),2)   
cv2.imshow("Gaussian blur",blur)
th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow("Adaptive thresholding",th3)
cv2.imshow("final",th3)
cv2.waitKey(0)
cv2.destroyAllWindows()