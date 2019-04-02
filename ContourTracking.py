import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

name = 'Ol'
i = 1
wl = 400

pfluor = cv.imread('Test\\'+name+str(i)+'_'+str(wl)+'.tiff', cv.IMREAD_GRAYSCALE)
p0 = cv.imread('Test\\'+name+str(i)+'_0.tiff', cv.IMREAD_GRAYSCALE)
p = pfluor-p0
ma1 = np.max(p)
pshow = (p*(255/ma1)).astype(np.uint8)
cv.namedWindow('picture', cv.WINDOW_NORMAL)
cv.imshow('picture', pshow)
cv.waitKey(0)

h = plt.hist(p.ravel(), 256)
plt.show()

m, thsimple = cv.threshold(p, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
print(m)
cv.namedWindow('thresh', cv.WINDOW_NORMAL)
cv.imshow('thresh', thsimple)
cv.waitKey(0)

blur = cv.GaussianBlur(p, (5, 5), 0)
plt.hist(blur.ravel(), 256)
plt.show()

m, thgauss = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
print(m)
cv.namedWindow('threshblur', cv.WINDOW_NORMAL)
cv.imshow('threshblur', thgauss)
cv.waitKey(0)
