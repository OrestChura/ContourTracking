import cv2 as cv
import numpy as np

def testtrack(picts, conts, num1, num2):

    cv.namedWindow('contoured', cv.WINDOW_NORMAL)
    cv.imshow('contoured', cv.drawContours(cv.cvtColor(picts[num2], cv.COLOR_GRAY2RGB), [conts[num2]], -1, (0, 255, 0), 3))
    cv.waitKey(0)

    nextPts, status, err = cv.calcOpticalFlowPyrLK(picts[num1], picts[num2],
                                                   np.float32([tr[-1] for tr in conts[num1]]).reshape(-1, 1, 2), None)

    prevPts, status, err = cv.calcOpticalFlowPyrLK(picts[num2], picts[num1],
                                                   np.float32([tr[-1] for tr in nextPts]).reshape(-1, 1, 2), None)

    m, diff = cv.threshold(abs((conts[num1] - prevPts)).reshape(conts[num1].shape[0], 2), 5, 1, cv.THRESH_BINARY)

    nPtcut = []
    for i in range(diff.shape[0]):
        if diff[i].max() == 0.:
            nPtcut.append(nextPts[i])

    cv.namedWindow('contoured_by', cv.WINDOW_NORMAL)
    cv.imshow('contoured_by', cv.drawContours(cv.cvtColor(picts[num2], cv.COLOR_GRAY2RGB),
                                              [np.int32(np.around(nPtcut))], -1, (0, 0, 255), 3))
    cv.waitKey(0)