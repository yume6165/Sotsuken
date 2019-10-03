#codeing:utf-8
# -*- coding: utf-8 -*-

import os, sys, time
import cv2 as cv
from PIL import Image
import numpy as np

def find_rect():#グラフカットのための長方形を決定するための関数
    global tmp_img#画像処理のための一時的な保管場所
    tmp_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    return 0


if __name__ == '__main__':
        img = cv.imread("./sample/incision_1.jpg")
        find_rect()
        cv.imshow("ウィンドウ",tmp_img)
        cv.waitKey()
