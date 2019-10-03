#codeing:utf-8
# -*- coding: utf-8 -*-

import os, sys, time
import cv2 as cv
from PIL import Image

if __name__ == '__main__':
        img = cv.imread("./sample/incision_1.jpg")
        cv.imshow("ウィンドウ",img)
        cv.waitKey()
