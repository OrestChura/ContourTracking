import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

p400 = cv.imread('Test\\1_400.tiff', cv.IMREAD_GRAYSCALE)
p0 = cv.imread('Test\\1_0.tiff', cv.IMREAD_GRAYSCALE)
p = p400-p0
ma1 = np.max(p)
psw = (p*(255/ma1)).astype(np.uint8)

# cv.namedWindow('400-0', cv.WINDOW_NORMAL)
# cv.imshow('400-0', psw)
# cv.waitKey(0)
#

############## #Thresholding/Binarization
# h = plt.hist(p.ravel(), 256)
# plt.show()

m, thresh = cv.threshold(p, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
print(m)
cv.namedWindow('thresh', cv.WINDOW_NORMAL)
cv.imshow('thresh', thresh)
cv.waitKey(0)

