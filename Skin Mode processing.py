#Code to generate skin mode processing images

# Required modules
import cv2
import numpy as np

# Constants for finding range of skin color in YCrCb
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)

sourceImage= cv2.imread("stop116.jpg")
cv2.imshow("original",sourceImage)

# Convert image to YCrCb
imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)
cv2.imshow("YCrCb space image",imageYCrCb)

# Find region with skin tone in YCrCb image
skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
cv2.imshow("Skin regions in image",skinRegion)

# Do contour detection on skin region
_,contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contour on the source image
for i, c in enumerate(contours):
   area = cv2.contourArea(c)
   if area > 1000:
      cv2.drawContours(sourceImage, contours, i, (0,0,0),thickness=2)


#Thresholding and gray-scaling the source image
sourceImage = cv2.cvtColor( sourceImage, cv2.COLOR_RGB2GRAY)
_, res = cv2.threshold(sourceImage, 0, 255, cv2.THRESH_BINARY)

# Display the source image
cv2.imshow('source image with skin regions',sourceImage)
cv2.imshow('thresholded source image',res)

cv2.waitKey(0)
cv2.destroyAllWindows()

