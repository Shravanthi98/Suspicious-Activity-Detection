import numpy as np
import cv2
def detect_tampering(fgbg, frame):
    kernel = np.ones((5, 5), np.uint8)
    area = 0
    boundRecParam = []
    fgmask = fgbg.apply(frame)
    fgmask = cv2.erode(fgmask, kernel, iterations=10)
    fgmask = cv2.dilate(fgmask, kernel, iterations=10)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        boundRecParam.append(cv2.boundingRect(contours[i]))
    for i in range(0, len(contours)):
        if boundRecParam[i][2] >= 30 or boundRecParam[i][3] >= 30:
            area = area + (boundRecParam[i][2]) * boundRecParam[i][3]
        if (area >= int(frame.shape[0]) * int(frame.shape[1]) / 3):
            return True
    return False
