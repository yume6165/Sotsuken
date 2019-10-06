#codeing:utf-8

import os, sys, time
import cv2 as cv
from PIL import Image
import numpy as np

#研究室で研究するとき
#path = "./sample/incision_1.jpg"

#ノートパソコンで研究するとき
path = "D:\Sotsuken\Sotsuken_repo./sample/incision_1.jpg"


def separate(img):#16区画に分ける
	global list_sepa, width, height
	global sepa1, sepa2, sepa3, sepa4, sepa5, sepa6, sepa7, sepa8
	global sepa9, sepa10, sepa11, sepa12, sepa13, sepa14, sepa15, sepa16
	#list_sepa = []
	width = int(img.shape[0] * 25 / 100)#１区画の幅の大きさ
	height = int(img.shape[1] * 25 / 100)#１区画の高さ
	
	sepa1 = img[(0)*width : 1* width, (0)*height: 1* height]
	sepa2 = img[(0)*width : 1* width, (1)*height: 2* height]
	sepa3 = img[(0)*width : 1* width, (2)*height: 3* height]
	sepa4 = img[(0)*width : 1* width, (3)*height: 4* height]
	sepa5 = img[(1)*width : 2* width, (0)*height: 1* height]
	sepa6 = img[(1)*width : 2* width, (1)*height: 2* height]
	sepa7 = img[(1)*width : 2* width, (2)*height: 3* height]
	sepa8 = img[(1)*width : 2* width, (3)*height: 4* height]
	sepa9 = img[(2)*width : 3* width, (0)*height: 1* height]
	sepa10 = img[(2)*width : 3* width, (1)*height: 2* height]
	sepa11 = img[(2)*width : 3* width, (2)*height: 3* height]
	sepa12 = img[(2)*width : 3* width, (3)*height: 4* height]
	sepa13 = img[(3)*width : 4* width, (0)*height: 1* height]
	sepa14 = img[(3)*width : 4* width, (1)*height: 2* height]
	sepa15 = img[(3)*width : 4* width, (2)*height: 3* height]
	sepa16 = img[(3)*width : 4* width, (3)*height: 4* height]
	
	cv.imshow("separate_"+ str(1), sepa1)
	#cv.imshow("separate_"+ str(2), img[(0)*width : 1* width, (1)*height: 2* height])
	#cv.imshow("separate_"+ str(3), img[(0)*width : 1* width, (2)*height: 3* height])
	#cv.imshow("separate_"+ str(4), img[(0)*width : 1* width, (3)*height: 4* height])
	#cv.imshow("separate_"+ str(5), img[(1)*width : 2* width, (0)*height: 1* height])
	#cv.imshow("separate_"+ str(6), img[(1)*width : 2* width, (1)*height: 2* height])
	#cv.imshow("separate_"+ str(7), img[(1)*width : 2* width, (2)*height: 3* height])
	#cv.imshow("separate_"+ str(8), img[(1)*width : 2* width, (3)*height: 4* height])
	#cv.imshow("separate_"+ str(9), img[(2)*width : 3* width, (0)*height: 1* height])
	#cv.imshow("separate_"+ str(10), img[(2)*width : 3* width, (1)*height: 2* height])
	#cv.imshow("separate_"+ str(11), img[(2)*width : 3* width, (2)*height: 3* height])
	#cv.imshow("separate_"+ str(12), img[(2)*width : 3* width, (3)*height: 4* height])
	#cv.imshow("separate_"+ str(13), img[(3)*width : 4* width, (0)*height: 1* height])
	#cv.imshow("separate_"+ str(14), img[(3)*width : 4* width, (1)*height: 2* height])
	#cv.imshow("separate_"+ str(15), img[(3)*width : 4* width, (2)*height: 3* height])
	#cv.imshow("separate_"+ str(16), img[(3)*width : 4* width, (3)*height: 4* height])
	
	#for文で回したいけどうまくいかない
	#for i in range(int(100/ rate)):
	#	for j in range(int(100/ rate)):
	#		cv.imshow("separate_"+ str(i), img[(i-1)*width : i* width, (j-1)*height: j* height])
	#		


def find_rect(img):#グラフカットのための長方形を決定するための関数
	global tmp_img, width, height#画像処理のための一時的な保管場所
	global tmp_img_re
	
	#パラメータ
	width = img.shape[0]
	height = img.shape[1]
	threshold = 110
	cut_rate = 80 #トリミングするときの比率
	comp_rate = 25 #4*4で画像を区切って
	
	print(str(width) + " " + str(height))
	
	
	tmp_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#グレースケールに変換
	
	#バイラテラルフィルタをかける
	for i in range(3):
		tmp_img = cv.bilateralFilter(tmp_img, 15, 20, 20)
		ret, tmp_img = cv.threshold(tmp_img, threshold, 255, cv.THRESH_BINARY)
	
	
	#傷の重心となる画素を探す
	
	#傷が写真の中心にあると仮定して真ん中のみを切り取り
	x1 = int(width * (100 - cut_rate) / 2 / 100)
	x2 = int(width * (1 - (100 - cut_rate) / 2 / 100))
	y1 = int(height * (100 - cut_rate) / 2 / 100)
	y2 = int(height * (1 - (100 - cut_rate) / 2 / 100))
	#print(str(x1)+ " " + str(x2) + ", " + str(y1) + " " + str(y2))
	
	tmp_img_re = tmp_img[x1 : x2, y1: y2]
	
	#トリミングした画素を16区画に分ける=sepa1~sepa16に画像のデータが入る
	separate(tmp_img_re)
	



if __name__ == '__main__':
		img = cv.imread(path)
		find_rect(img)
		cv.imshow("incision1",img)
		cv.imshow("incision2",tmp_img)
		cv.waitKey()
