import os, sys, time
import cv2 as cv
from PIL import Image, ImageTk

if __name__ == '__main__':
        image = cv.imread(".\sample\incision_1.jpg")
        cv.imshow(image)
