import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

name = 'T'
# i = 1
wl = 400

picts = []
conts = []

for i in [1, 2]:
    pfluor = cv.imread('Test\\' + name + str(i) + '_' + str(wl) + '.tiff', cv.IMREAD_GRAYSCALE)
    p0 = cv.imread('Test\\' + name + str(i) + '_0.tiff', cv.IMREAD_GRAYSCALE)
    p = pfluor - p0
    ma1 = np.max(p)
    pshow = (p * (255 / ma1)).astype(np.uint8)
    cv.namedWindow('picture', cv.WINDOW_NORMAL)
    cv.imshow('picture', pshow)
    cv.waitKey(0)

    # h = plt.hist(p.ravel(), 256)
    # plt.show()
    #
    # m, thsimple = cv.threshold(p, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # print(m)
    # cv.namedWindow('thresh', cv.WINDOW_NORMAL)
    # cv.imshow('thresh', thsimple)
    # cv.waitKey(0)

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

    blur = cv.GaussianBlur(p, (9, 9), 0)
    blurshow = (blur * (255 / ma1)).astype(np.uint8)
    cv.namedWindow('picture1', cv.WINDOW_NORMAL)
    cv.imshow('picture1', blurshow)
    cv.waitKey(0)
    # plt.hist(blur.ravel(), 256)
    # plt.show()

    m, thgauss = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    print(m)
    cv.namedWindow('threshblur', cv.WINDOW_NORMAL)
    cv.imshow('threshblur', thgauss)
    cv.waitKey(0)

    picts.append(thgauss)

    cont, hier = cv.findContours(thgauss, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    conts = conts + cont

    cv.namedWindow('contoured', cv.WINDOW_NORMAL)
    cv.imshow('contoured', cv.drawContours(cv.cvtColor(thgauss, cv.COLOR_GRAY2RGB), cont, -1, (0, 255, 0), 2))
    cv.waitKey(0)

# TODO: возможно, исп. эту ф-ю
# cv.goodFeaturesToTrack() - функция для отыскания углов

nextPts, status, err = cv.calcOpticalFlowPyrLK(picts[0], picts[1],
                                               np.float32([tr[-1] for tr in conts[0]]).reshape(-1, 1, 2), None)
cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(thgauss, cv.COLOR_GRAY2RGB),
                                          [np.int32(np.around(nextPts))], -1, (0, 0, 255), 2))
cv.waitKey(0)
