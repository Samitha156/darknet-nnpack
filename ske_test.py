import cv2
import numpy as np
from skimage.morphology import skeletonize

print("checking Skel")
same_frame = cv2.imread("neu.jpg")
same_frame = cv2.blur(same_frame, (5, 5))
imgray = cv2.cvtColor(same_frame, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(imgray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
sk = skeletonize(thresh)
ske = np.asarray(sk) *255
cv2.imwrite("skel.jpg",ske)