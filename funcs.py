import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import tkinter.messagebox as mb
from tkinter import ttk


# picts(raw), threshs, conts = pictsconts()
def pictsconts(name, wl, begin, end):
    picts = []
    threshs = []
    conts = []
    # root = tkinter.Tk()
    # root.title('Loading pictures...')
    # bar = ttk.Progressbar(master=root, orient='horizontal', length=300, mode='determinate', maximum=end, value=0)
    # bar.pack()
    #
    # def fbar():
    #     nonlocal picts
    #     nonlocal threshs
    #     nonlocal conts
    #     cv.namedWindow('picture', cv.WINDOW_NORMAL)
    for i in range(begin, end + 1):
        pfluor = cv.imread('Test\\' + name + str(i) + '_' + str(wl) + '.tiff', cv.IMREAD_GRAYSCALE)
        p0 = cv.imread('Test\\' + name + str(i) + '_0.tiff', cv.IMREAD_GRAYSCALE)
        p = pfluor - p0
        ma1 = np.max(p)

        if ma1 > 252:
            m, p = cv.threshold(p, 252, 255, cv.THRESH_TOZERO_INV)

        blur = cv.GaussianBlur(p, (5, 5), 0)
        picts.append(blur)
        # blurshow = (blur * (255 / np.max(blur))).astype(np.uint8)
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
        print(str(i)+'/'+str(end))
    #     bar['value'] = i
    #     bar.update_idletasks()
    # root.destroy()
    # b = tkinter.Button(root, text='Begin', command=fbar)
    # b.pack(side='bottom')
    # root.lift()
    # root.mainloop()
    return picts, threshs, conts


# None = shppc()
def showrawpicts(name, wl, begin, end):
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
# def tracktwopicts(picts, conts, num1, num2=None):
#     if num2 is None:
#         num2 = num1 + 1
#
#     # cv.namedWindow('contoured', cv.WINDOW_NORMAL)
#     # cv.imshow('contoured', cv.drawContours(cv.cvtColor(picts[num2], cv.COLOR_GRAY2RGB),
#     #                                        list(conts[num2]), -1, (0, 255, 0), 3))
#     # cv.waitKey(0)
#
#     nextpts, status, err = cv.calcOpticalFlowPyrLK(picts[num1], picts[num2],
#                                                    np.float32([tr[-1] for tr in conts[num1]]).reshape(-1, 1, 2), None)
#
#     prevpts, status, err = cv.calcOpticalFlowPyrLK(picts[num2], picts[num1],
#                                                    np.float32([tr[-1] for tr in nextpts]).reshape(-1, 1, 2), None)
#
#     m, diff = cv.threshold(abs((conts[num1] - prevpts)).reshape(-1, 2), 5, 1, cv.THRESH_BINARY)
#
#     nptcut = []
#     for i in range(diff.shape[0]):
#         if diff[i].max() == 0.:
#             nptcut.append(nextpts[i])
#
#     cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
#     cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(picts[num2], cv.COLOR_GRAY2RGB),
#                                               list(np.int32(np.around(nptcut))), -1, (0, 0, 255), 3))
#     cv.waitKey(0)


# None = ()
# на каждом шаге выводит количество оставшихся точек

# cont = tr2p()
def tracktwopicts(prevthresh, prevcont, nextthresh,  wsize, maxlvl, delta):
    nextpts, status, err = cv.calcOpticalFlowPyrLK(prevthresh, nextthresh,
                                                   np.float32([tr[-1] for tr in prevcont]).reshape(-1, 1, 2),
                                                   None, winSize=wsize, maxLevel=maxlvl)

    prevpts, status, err = cv.calcOpticalFlowPyrLK(nextthresh, prevthresh,
                                                   np.float32([tr[-1] for tr in nextpts]).reshape(-1, 1, 2),
                                                   np.float32([tr[-1] for tr in prevcont]).reshape(-1, 1, 2),
                                                   winSize=wsize, maxLevel=maxlvl)

    m, diff = cv.threshold(
        np.array([[(i ** 2 + j ** 2) ** (1 / 2)] for [i, j] in abs(prevcont - prevpts).reshape(-1, 2)]),
        delta, 1, cv.THRESH_BINARY_INV)
    return np.around([i for (i, j) in zip(nextpts.reshape(-1, 2), diff) if j]).astype(np.int32).reshape((-1, 1, 2))


# None = ()
# def trackseries_n_compare(picts, threshs, conts):
def trackseries_n_compare(picts, threshs, wsize, maxlvl, delta, compare=False):
    showpicts = [(p * (255 / np.max(p))).astype(np.uint8) for p in picts]
    # cv.namedWindow('contoured', cv.WINDOW_NORMAL)
    # cv.imshow('contoured', cv.drawContours(cv.cvtColor(showpicts[0], cv.COLOR_GRAY2RGB),
    #                                        list(conts[0]), -1, (0, 255, 0), 3))
    # cv.waitKey(0)

    # contby = conts[0]
    contby = safecont = firstcont = np.array(manual_contour((showpicts[0]))).reshape((-1, 1, 2))

    cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
    cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(showpicts[0], cv.COLOR_GRAY2RGB),
                                              [contby], -1, (0, 0, 255), 3))
    cv.waitKey(0)
    n_bad_pictures = 0
    for i in range(1, len(threshs) - 1):
        # cv.setWindowTitle('contoured', 'contoured' + str(i + 1))
        # cv.imshow('contoured', cv.drawContours(cv.cvtColor(showpicts[i], cv.COLOR_GRAY2RGB),
        #                                        list(conts[i]), -1, (0, 255, 0), 3))
        # cv.waitKey(0)
        newcontby = tracktwopicts(threshs[i - n_bad_pictures - 1], contby, threshs[i], wsize, maxlvl, delta)
        # newcontby = np.around(nextpts).astype(np.int32)
        print(str(i + 1) + ': ' + str(newcontby.shape[0]))
        cv.setWindowTitle('contoured_by', 'contoured_by' + str(i + 1))

        if contby.size <= newcontby.size + 10:
            cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(showpicts[i], cv.COLOR_GRAY2RGB),
                                                      [newcontby], -1, (0, 0, 255), 3))
            cv.waitKey(0)
            safecont, contby = contby, newcontby
            n_bad_pictures = 0
        else:
            if n_bad_pictures < 4:
                cv.imshow('contoured_by', showpicts[i])
                cv.waitKey(0)
                n_bad_pictures = n_bad_pictures + 1
            else:
                newcontby = tracktwopicts(threshs[i - n_bad_pictures - 2], safecont, threshs[i], wsize, maxlvl, delta)
                if contby.size <= newcontby.size + 10:
                    cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(showpicts[i], cv.COLOR_GRAY2RGB),
                                                              [newcontby], -1, (0, 0, 255), 3))
                    cv.waitKey(0)
                    contby = newcontby
                    n_bad_pictures = 0
                else:
                    root = tkinter.Tk()
                    root.withdraw()
                    ans = mb.showerror('Ошибка', 'Контур потерялся. Пожалуйста, введите новый', parent=root)
                    root.destroy()
                    contby = safecont = np.array(manual_contour((showpicts[i]))).reshape((-1, 1, 2))

                    cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
                    cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(showpicts[0], cv.COLOR_GRAY2RGB),
                                                              [contby], -1, (0, 0, 255), 3))
                    cv.waitKey(0)
                    n_bad_pictures = 0
        # try:
        #     cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(showpicts[i], cv.COLOR_GRAY2RGB),
        #                                               [newcontby], -1, (0, 0, 255), 3))
        #     cv.waitKey(0)
        #     contby = newcontby
        #     n_bad_pictures = 0
        # except Exception:
        #     root = tkinter.Tk()
        #     root.withdraw()
        #     ans = mb.showerror('Ошибка', 'Все точки потерялись.', parent=root)
        #     root.destroy()
        #     cv.imshow('contoured_by', showpicts[i])
        #     cv.waitKey(0)
        #     n_bad_pictures = n_bad_pictures + 1
    if compare:
        cv.namedWindow('firstpicture', cv.WINDOW_NORMAL)
        cv.imshow('firstpicture', cv.drawContours(cv.cvtColor(showpicts[0], cv.COLOR_GRAY2RGB),
                                                  [firstcont], -1, (0, 0, 255), 3))
        cv.waitKey(0)


# poly = [[x1,y1],...] = man_con()
def manual_contour(pict):
    cv.namedWindow('input', cv.WINDOW_NORMAL)
    cv.moveWindow('input', 1, 1)

    manual_input_parameters = {"color": (0, 255, 0),
                               "pict": pict.copy(),
                               "pict_normalized": cv.cvtColor(pict, cv.COLOR_GRAY2RGB),
                               "poly": [],
                               # setting up flags
                               "is_poly_drawing": False,
                               "is_poly_over": False,
                               "thickness": 3}

    manual_input_parameters["pict_normalized"] = (manual_input_parameters["pict_normalized"].astype(np.float)
                                                  / np.max(manual_input_parameters["pict_normalized"])
                                                  * 255).astype(np.uint8)
    cv.imshow('input', manual_input_parameters["pict_normalized"])

    def onmouse(event, x, y, flags, param):
        nonlocal manual_input_parameters
        # Draw Rectangle
        if event == cv.EVENT_LBUTTONDOWN:
            manual_input_parameters["is_poly_drawing"] = True
            manual_input_parameters["is_poly_over"] = False
            manual_input_parameters["poly"] = manual_input_parameters["poly"] + [[x, y]]
        elif event == cv.EVENT_MOUSEMOVE:
            if manual_input_parameters["is_poly_drawing"] is True:
                manual_input_parameters["pict"] = manual_input_parameters["pict_normalized"].copy()
                manual_input_parameters["poly"] = manual_input_parameters["poly"] + [[x, y]]
                cv.polylines(manual_input_parameters["pict"], [np.array(manual_input_parameters["poly"], np.int32)],
                             False, manual_input_parameters["color"], manual_input_parameters["thickness"])
                cv.imshow('input', manual_input_parameters["pict"])
        elif event == cv.EVENT_LBUTTONUP:
            manual_input_parameters["is_poly_drawing"] = False
            manual_input_parameters["is_poly_over"] = True
            manual_input_parameters["poly"] = manual_input_parameters["poly"] + [[x, y]]

            cv.polylines(manual_input_parameters["pict"],
                         [np.array(manual_input_parameters["poly"], np.int32)], True,
                         manual_input_parameters["color"], manual_input_parameters["thickness"])

            cv.imshow('input', manual_input_parameters["pict"])
        elif event == cv.EVENT_RBUTTONUP:
            manual_input_parameters["poly"] = []
            manual_input_parameters["is_poly_over"] = False
            cv.imshow('input', manual_input_parameters["pict_normalized"])

    cv.setMouseCallback('input', onmouse)
    cv.waitKey(0)
    cv.destroyWindow('input')
    return manual_input_parameters["poly"]
