import tkinter
import tkinter.messagebox as mb

from funcs import *

picts, threshs, conts = pictsconts('T', 400, 1, 43)

cv.namedWindow('contoured', cv.WINDOW_NORMAL)
cv.imshow('contoured', cv.drawContours(cv.cvtColor(threshs[0], cv.COLOR_GRAY2RGB),
                                       list(conts[0]), -1, (0, 255, 0), 3))
cv.waitKey(0)

cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(threshs[0], cv.COLOR_GRAY2RGB),
                                          list(conts[0]), -1, (0, 0, 255), 3))
cv.waitKey(0)

contby = conts[0]
print(str(1)+': '+str(contby.shape[0]))
k = 0
for i in range(1, len(threshs) - 1):
    cv.setWindowTitle('contoured', 'contoured' + str(i+1))
    cv.imshow('contoured', cv.drawContours(cv.cvtColor(threshs[i], cv.COLOR_GRAY2RGB),
                                           list(conts[i]), -1, (0, 255, 0), 3))
    cv.waitKey(0)

    nextpts, status, err = cv.calcOpticalFlowPyrLK(threshs[i-k-1], threshs[i],
                                                   np.float32([tr[-1] for tr in contby]).reshape(-1, 1, 2),
                                                   None, maxLevel=6)

    prevpts, status, err = cv.calcOpticalFlowPyrLK(threshs[i], threshs[i-k-1],
                                                   np.float32([tr[-1] for tr in nextpts]).reshape(-1, 1, 2),
                                                   np.float32([tr[-1] for tr in contby]).reshape(-1, 1, 2), maxLevel=6)
    # TODO: найти оптимальный порог выкидывания точек
    m, diff = cv.threshold(abs((contby - prevpts)).reshape(contby.shape[0], 2), 5, 1, cv.THRESH_BINARY)
    # TODO: найти способ сделать это с помощью numpy
    ######
    nptcut = []
    for j in range(diff.shape[0]):
        if diff[j].max() == 0.:
            nptcut.append(nextpts[j])
    contby1 = np.int32(np.around(nptcut))
    ######
    print(str(i + 1) + ': ' + str(contby1.shape[0]))
    cv.setWindowTitle('contoured_by', 'contoured_by' + str(i + 1))
    try:
        cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(threshs[i], cv.COLOR_GRAY2RGB),
                                                  [contby], -1, (0, 0, 255), 3))
        cv.waitKey(0)
        contby = contby1
        k = 0
    except Exception:
        root = tkinter.Tk()
        ans = mb.showerror('Ошибка', 'Все точки потерялись.',
                           parent=root)
        root.destroy()
        cv.imshow('contoured_by', threshs[i])
        cv.waitKey(0)
        k = k + 1

# TODO: возможно, использ. эту ф-ю
# cv.goodFeaturesToTrack() - функция для отыскания углов

# nextPts, status, err = cv.calcOpticalFlowPyrLK(threshs[0], threshs[1],
#                                                np.float32([tr[-1] for tr in conts[0]]).reshape(-1, 1, 2), None)
# cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
# cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(thgauss, cv.COLOR_GRAY2RGB),
#                                           [np.int32(np.around(nextPts))], -1, (0, 0, 255), 3))
# cv.waitKey(0)
pass
# TODO: найти оптимальный Гаусс
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# name = 'T'
# # i = 1
# wl = 400
#
# picts = []
# conts = []
#
# i = 1
#
# pfluor = cv.imread('Test\\' + name + str(i) + '_' + str(wl) + '.tiff', cv.IMREAD_GRAYSCALE)
# p0 = cv.imread('Test\\' + name + str(i) + '_0.tiff', cv.IMREAD_GRAYSCALE)
# p = pfluor - p0
# ma1 = np.max(p)
# pshow = (p * (255 / ma1)).astype(np.uint8)
# cv.namedWindow('picture', cv.WINDOW_NORMAL)
# cv.imshow('picture', pshow)
# cv.waitKey(0)
#
# cv.namedWindow('picture1', cv.WINDOW_NORMAL)
# for i in (range(10)):
#     blur = cv.GaussianBlur(p, (i * 2 + 1, i * 2 + 1), 0)
#     blurshow = (blur * (255 / ma1)).astype(np.uint8)
#
#     cv.imshow('picture1', blurshow)
#     cv.waitKey(0)
