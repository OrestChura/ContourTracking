import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import tkinter.messagebox as mb


# picts, threshs, conts = pictsconts()
def pictsconts(name, wl, begin, end):
    picts = []
    threshs = []
    conts = []
    # cv.namedWindow('picture', cv.WINDOW_NORMAL)
    for i in range(begin, end + 1):
        pfluor = cv.imread('Test\\' + name + str(i) + '_' + str(wl) + '.tiff', cv.IMREAD_GRAYSCALE)
        p0 = cv.imread('Test\\' + name + str(i) + '_0.tiff', cv.IMREAD_GRAYSCALE)
        p = pfluor - p0
        ma1 = np.max(p)

        if ma1 > 252:
            m, p = cv.threshold(p, 252, 255, cv.THRESH_TOZERO_INV)

        blur = cv.GaussianBlur(p, (5, 5), 0)
        picts.append(blur)
        # blurshow = (blur * (255 / ma1)).astype(np.uint8)
        # cv.namedWindow('picture1', cv.WINDOW_NORMAL)
        # cv.imshow('picture1', blurshow)
        # cv.waitKey(0)
        # plt.hist(blur.ravel(), 256)
        # plt.show()

        m, thgauss = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # print(m)
        # cv.namedWindow('threshblur', cv.WINDOW_NORMAL)
        # cv.imshow('threshblur', thgauss)
        # cv.waitKey(0)

        threshs.append(thgauss)

        cont, hier = cv.findContours(thgauss, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        conts = conts + cont

        # cv.namedWindow('contoured', cv.WINDOW_NORMAL)
        # cv.imshow('contoured', cv.drawContours(cv.cvtColor(thgauss, cv.COLOR_GRAY2RGB),
        #                                        list(cont), -1, (0, 255, 0), 3))
        # cv.waitKey(0)
    return picts, threshs, conts


# None = shppc()
def showpurepicts(name, wl, begin, end):
    cv.namedWindow('picture', cv.WINDOW_NORMAL)
    for i in range(begin, end + 1):
        pfluor = cv.imread('Test\\' + name + str(i) + '_' + str(wl) + '.tiff', cv.IMREAD_GRAYSCALE)
        p0 = cv.imread('Test\\' + name + str(i) + '_0.tiff', cv.IMREAD_GRAYSCALE)
        p = pfluor - p0
        ma1 = np.max(p)

        if ma1 > 252:
            m, p = cv.threshold(p, 252, 255, cv.THRESH_TOZERO_INV)
            ma1 = np.max(p)

        pshow = (p * (255 / ma1)).astype(np.uint8)
        cv.imshow('picture', pshow)
        cv.setWindowTitle('picture', 'picture' + str(i))
        cv.waitKey(0)

        # h = plt.hist(p.ravel(), 256)
        # plt.show()


# None = ()
def tracktwopicts(picts, conts, num1, num2=None):
    if num2 is None:
        num2 = num1 + 1

    cv.namedWindow('contoured', cv.WINDOW_NORMAL)
    cv.imshow('contoured', cv.drawContours(cv.cvtColor(picts[num2], cv.COLOR_GRAY2RGB),
                                           list(conts[num2]), -1, (0, 255, 0), 3))
    cv.waitKey(0)

    nextpts, status, err = cv.calcOpticalFlowPyrLK(picts[num1], picts[num2],
                                                   np.float32([tr[-1] for tr in conts[num1]]).reshape(-1, 1, 2), None)

    prevpts, status, err = cv.calcOpticalFlowPyrLK(picts[num2], picts[num1],
                                                   np.float32([tr[-1] for tr in nextpts]).reshape(-1, 1, 2), None)

    m, diff = cv.threshold(abs((conts[num1] - prevpts)).reshape(-1, 2), 5, 1, cv.THRESH_BINARY)

    nptcut = []
    for i in range(diff.shape[0]):
        if diff[i].max() == 0.:
            nptcut.append(nextpts[i])

    cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
    cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(picts[num2], cv.COLOR_GRAY2RGB),
                                              list(np.int32(np.around(nptcut))), -1, (0, 0, 255), 3))
    cv.waitKey(0)


#None = ()
def trackseries(picts, threshs, conts):
    cv.namedWindow('contoured', cv.WINDOW_NORMAL)
    cv.imshow('contoured', cv.drawContours(cv.cvtColor(threshs[0], cv.COLOR_GRAY2RGB),
                                           list(conts[0]), -1, (0, 255, 0), 3))
    cv.waitKey(0)

    cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
    cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(threshs[0], cv.COLOR_GRAY2RGB),
                                              list(conts[0]), -1, (0, 0, 255), 3))
    cv.waitKey(0)

    contby = conts[0]
    print(str(1) + ': ' + str(contby.shape[0]))
    k = 0
    for i in range(1, len(threshs) - 1):
        cv.setWindowTitle('contoured', 'contoured' + str(i + 1))
        cv.imshow('contoured', cv.drawContours(cv.cvtColor(threshs[i], cv.COLOR_GRAY2RGB),
                                               list(conts[i]), -1, (0, 255, 0), 3))
        cv.waitKey(0)

        nextpts, status, err = cv.calcOpticalFlowPyrLK(threshs[i - k - 1], threshs[i],
                                                       np.float32([tr[-1] for tr in contby]).reshape(-1, 1, 2),
                                                       None, maxLevel=6)

        prevpts, status, err = cv.calcOpticalFlowPyrLK(threshs[i], threshs[i - k - 1],
                                                       np.float32([tr[-1] for tr in nextpts]).reshape(-1, 1, 2),
                                                       np.float32([tr[-1] for tr in contby]).reshape(-1, 1, 2),
                                                       maxLevel=6)
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
                                                      [contby1], -1, (0, 0, 255), 3))
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