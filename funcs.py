import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tkinter
import tkinter.messagebox as mb

BTN_ESC = 27

# from tkinter import ttk


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
        print(str(i) + '/' + str(end))
    #     bar['value'] = i
    #     bar.update_idletasks()
    # root.destroy()
    # b = tkinter.Button(root, text='Begin', command=fbar)
    # b.pack(side='bottom')
    # root.lift()
    # root.mainloop()
    return picts, threshs, conts


# None = shrpc()
def showrawpicts(name, wl, begin, end):
    cv.namedWindow('picture', cv.WINDOW_NORMAL)
    for i in range(begin, end + 1):
        pfluor = cv.imread('Test\\' + name + str(i) + '_' + str(wl) + '.tiff', cv.IMREAD_GRAYSCALE)
        p0 = cv.imread('Test\\' + name + str(i) + '_0.tiff', cv.IMREAD_GRAYSCALE)
        p = pfluor - p0
        ma1 = np.max(p)

        # if ma1 > 252:
        #     m, p = cv.threshold(p, 252, 255, cv.THRESH_TOZERO_INV)
        #     ma1 = np.max(p)

        pshow = (p * (255 / ma1)).astype(np.uint8)
        cv.imshow('picture', pshow)
        cv.setWindowTitle('picture', 'picture' + str(i))

        # h = plt.hist(p.ravel(), 256)
        # plt.show()

        if cv.waitKey(0) == BTN_ESC:
            cv.destroyAllWindows()
            break


# None = shp()
def showpicts(name, wl, begin, end):
    cv.namedWindow('picture', cv.WINDOW_NORMAL)
    for i in range(begin, end + 1):
        pfluor = cv.imread('Test\\' + name + str(i) + '_' + str(wl) + '.tiff', cv.IMREAD_GRAYSCALE)
        p0 = cv.imread('Test\\' + name + str(i) + '_0.tiff', cv.IMREAD_GRAYSCALE)
        p = pfluor - p0
        ma1 = np.max(p)

        if ma1 > 252:
            m, p = cv.threshold(p, 252, 255, cv.THRESH_TOZERO_INV)
            ma1 = np.max(p)

        blur = cv.GaussianBlur(p, (5, 5), 0)
        pshow = (blur * (255 / ma1)).astype(np.uint8)
        cv.imshow('picture', pshow)
        cv.setWindowTitle('picture', 'picture' + str(i))

        # h = cv.calcHist(blur, [0], None, [255], [[0, 255]])

        m, thgauss = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        print(m)
        cv.namedWindow('threshblur', cv.WINDOW_NORMAL)
        cv.imshow('threshblur', thgauss)
        if cv.waitKey(0) == BTN_ESC:
            cv.destroyAllWindows()
            break



# cont = tr2p()
def tracktwopicts(prevthresh, prevcont, nextthresh, wsize, maxlvl, delta):
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
# на каждом шаге выводит количество оставшихся точек, n_i, n_sum
def trackseries_n_compare(wsize, maxlvl, delta, max_bad_pictures, n_dots_out, name, wl, end, begin=1,
                          compare=False, ):
    picts, threshs, conts = pictsconts(name, wl, begin, end)
    shwpicts = [(p * (255 / np.max(p))).astype(np.uint8) for p in picts]

    cv.namedWindow('thresh', cv.WINDOW_NORMAL)
    cv.imshow('thresh', threshs[0])

    contby = safecont = firstcont = np.array(manual_contour((shwpicts[0]))).reshape((-1, 1, 2))
    cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
    imgtoshow = cv.drawContours(cv.cvtColor(shwpicts[0], cv.COLOR_GRAY2RGB),
                                [contby], -1, (0, 0, 255), 3)
    cv.imshow('contoured_by', imgtoshow)
    if cv.waitKey(0) == 3:
        cv.imwrite('Output\\' + name + str(1) + '_' + str(wl) + '.tiff', imgtoshow)
        cv.waitKey(0)
    n_bad_pictures = n_laz_pictures = add_because_lazer = 0
    n_sum = 0
    for i in range(1, len(threshs)):
        cv.setWindowTitle('contoured_by', 'contoured_by' + str(i + 1))
        
        cv.setWindowTitle('thresh', 'thresh' + str(i + 1))
        cv.imshow('thresh', threshs[i])

        if np.sum(cv.calcHist([picts[i]], [0], None, [256], [0, 256])[230:]) < 30:
            newcontby = tracktwopicts(threshs[i - n_bad_pictures - n_laz_pictures - 1], contby, threshs[i],
                                      wsize, maxlvl, delta)
            n_i = (contby.shape[0] - newcontby.shape[0]) * 100. / contby.shape[0]
            if contby.size <= newcontby.size + n_dots_out*2:
                imgtoshow = cv.drawContours(cv.cvtColor(shwpicts[i], cv.COLOR_GRAY2RGB),
                                            [newcontby], -1, (0, 0, 255), 3)
                safecont, contby = contby, newcontby
                n_bad_pictures = n_laz_pictures = add_because_lazer = 0
                n_sum += n_i
            else:
                imgtoshow = shwpicts[i]
                n_bad_pictures = n_bad_pictures + 1
            print(str(i + 1) + ': N_i = ' + str(newcontby.shape[0]) + ', n_i = ' + str(n_i) + ', n_s = ' + str(n_sum))
        else:
            imgtoshow = shwpicts[i]
            n_laz_pictures = n_laz_pictures + 1
            if not add_because_lazer and n_bad_pictures:
                add_because_lazer = n_bad_pictures

        cv.imshow('contoured_by', imgtoshow)
        ans = cv.waitKey(0)
        if ans == BTN_ESC:
            break
        elif ans == 3:
            cv.imwrite('Output\\' + name + str(i + 1) + '_' + str(wl) + '.tiff', imgtoshow)
            if cv.waitKey(0) == BTN_ESC:
                break
    if compare:
        cv.namedWindow('firstpicture', cv.WINDOW_NORMAL)
        cv.imshow('firstpicture', cv.drawContours(cv.cvtColor(shwpicts[0], cv.COLOR_GRAY2RGB),
                                                  [firstcont], -1, (0, 0, 255), 3))
        if cv.waitKey(0) == 3:
            cv.imwrite('Output\\' + name + str(1) + '_' + str(wl) + '.tiff', imgtoshow)


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
