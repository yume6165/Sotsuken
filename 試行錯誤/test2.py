#codeing:utf-8
# -*- coding: utf-8 -*-

import os, sys, time
import cv2 as cv
from PIL import Image
import numpy as np
import math

#研究室で研究するとき
#path = "./sample/incision_1.jpg"

#ノートパソコンで研究するとき
path = "D:\Sotsuken\Sotsuken_repo./sample/incision_1.jpg"

N = 1000


def find_gravity_r(img):#HSVカラーモデルから重心を探す
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
	h = hsv[:, :, 0]#色相(赤の範囲は256段階の200～20と定義するfromhttps://qiita.com/odaman68000/items/ae28cf7bdaf4fa13a65b)
	s = hsv[:, :, 1]
	mask = np.zeros(h.shape, dtype=np.uint8)
	mask[((h < 20) | (h > 200)) & (s > 128)] = 255
	
	#輪郭を作るうえで塊ごとに配列化する
	contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	rects = []
	for contour in contours:#contourには輪郭をなすピクセルの情報が入っている
		approx = cv.convexHull(contour)#凸凹のある塊を内包する凸上の形状を算出して２Dポリゴンに
		rect = cv.boundingRect(approx)#袋状になったポリゴンがすっぽり入る四角を計算する
		rects.append(np.array(rect))
		
	#for rect in rects:
		#cv.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (0, 0, 255), thickness=2)
		#cv.imshow('red', img)
		
	#最大の四角を見つける
	result_rect = max(rects, key=(lambda x: x[2] * x[3]))
			
	cv.rectangle(img, tuple(result_rect[0:2]), tuple(result_rect[0:2] + result_rect[2:4]), (0, 255, 0), thickness=2)
	print(result_rect)
	re_img = img[ result_rect[1] : result_rect[1] + result_rect[3], result_rect[0] : result_rect[0] + result_rect[2]]
	x = result_rect[0] + int(round(result_rect[2]/2))
	y = result_rect[1] + int(round(result_rect[3]/2))
	cv.imshow('re_red', re_img)
	g_point = np.array([x, y])
	
	return g_point

def equal_list(lst1, lst2):
    lst = list(str(lst1))
    for element in lst2:
        try:
            lst.remove(element)
        except ValueError:
            break
    else:
        if not lst:
            return True
    return False


def review(img, weight):#16区画の評価(2値化された画像を想定)+重みづけ
	point = 0
	tmp_array = img.flatten()#一次元配列に変換

	for pixel in tmp_array:
		#print(pixel)
		if( pixel == 0 ):
			point += 1

	return point * weight

def separate(img):#16区画に分ける
	global list_sepa, width, height
	global sepa1, sepa2, sepa3, sepa4, sepa5, sepa6, sepa7, sepa8
	global sepa9, sepa10, sepa11, sepa12, sepa13, sepa14, sepa15, sepa16
	list_sepa = []
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

	list_sepa.append(sepa1)
	list_sepa.append(sepa2)
	list_sepa.append(sepa3)
	list_sepa.append(sepa4)
	list_sepa.append(sepa5)
	list_sepa.append(sepa6)
	list_sepa.append(sepa7)
	list_sepa.append(sepa8)
	list_sepa.append(sepa9)
	list_sepa.append(sepa10)
	list_sepa.append(sepa11)
	list_sepa.append(sepa12)
	list_sepa.append(sepa13)
	list_sepa.append(sepa14)
	list_sepa.append(sepa15)
	list_sepa.append(sepa16)


	#cv.imshow("separate_"+ str(10), sepa10)
	#cv.imshow("separate_"+ str(2), sepa2)
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

def img_gfilter(img):#重心検索用
	tmp_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#グレースケールに変換
	tmp_img = cv.GaussianBlur(tmp_img, (5, 5), 3)#ガウシアンフィルタ
	tmp_img = cv.GaussianBlur(tmp_img, (5, 5), 3)#ガウシアンフィルタ
	tmp_img = cv.bilateralFilter(tmp_img, 15, 20, 20)#バイラテラルフィルタをかける
	tmp_img = cv.GaussianBlur(tmp_img, (5, 5), 3)#ガウシアンフィルタ
	return tmp_img
	
def find_gravity(img):#グラフカットをもちいた重心の検索
	#スレッショルド設定
	THRESH_MIN, THRESH_MAX = (160, 255)
	TRESH_MODE = cv.THRESH_BINARY_INV
	
	#フィルター設定
	AA = (1001, 1001)
	CONTRAST = 1.1
	SHARPNESS = 1.5
	BRIGHTNESS = 1.1
	SATURATION = 1.0
	GAMMA = 1.0
	
	#グラフカット処理
	img_gray = img_gfilter(img)
	img_bin = cv.threshold(img_gray, THRESH_MIN, THRESH_MAX, TRESH_MODE)[1]
	
	#二値化画像からマスク画像を生成
	img_mask = cv.merge((img_bin, img_bin, img_bin))
	
	#マスク画像からshapeで矩形を囲い、その座標を取得
	mask_rows, mask_cols, mask_channel = img_mask.shape
	min_x = mask_cols
	min_y = mask_rows
	max_x = 0
	max_y = 0
	
	for y in range(mask_rows):
		for x in range(mask_cols):
			if all(img_mask[y, x] == 255):
				if x < min_x:
					min_x = x
				elif x > max_x:
					max_x = x
				if y < min_y:
					min_y = y
				elif y > max_y:
					max_y = y
	
	rect_x = min_x
	rect_y = min_y
	rect_w = max_x - min_x
	rect_h = max_y - min_y
	
	#前提マスクデータ格納準備
	mask = np.zeros(img.shape[:2], np.uint8)
	
	#前景領域データ、背景領域データ格納準備
	bg_model = np.zeros((1, 65), np.float64)
	fg_model = np.zeros((1, 65), np.float64)
	
	#前景画像の矩形領域設定
	rect = (int(img.shape[1]/4), int(img.shape[0]/3),  img.shape[1] - int(img.shape[1]/3), img.shape[0] - int(img.shape[0]/4))
	
	#矩形グラフカットデータ化
	cv.grabCut(img, mask, rect, bg_model, fg_model, 5, cv.GC_INIT_WITH_RECT)
	
	#領域分割
	#0:矩形外→０
	#1:矩形内（グラフカットによる背景かもしれない領域）→０
	#2:矩形内（前景確定）→１
	#3:矩形内（グラフカットによる前景かもしれない領域）→１
	
	mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
	
	#グラフカット処理
	img_grab = img * mask2[:, :, np.newaxis]
	
	cv.imshow("gray", img_bin)
	cv.imshow("grab", img_grab)
	
	
	return 0


def find_gravity_b(img):#傷の重心を探す関数
	global tmp_img, width, height, x1, x2, y1, y2, N#画像処理のための一時的な保管場所
	global tmp_img_re, list_sepa, binary_image#２値画像
	global point_dict, point1, point2, g_point, tmp_x, tmp_y

	#パラメータ
	width = img.shape[0]
	height = img.shape[1]
	threshold = 110
	cut_rate = 70 #トリミングするときの比率
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
	x2 = int(width - x1)
	y1 = int(height * (100 - cut_rate) / 2 / 100)
	y2 = int(height - y1)
	#print(str(x1)+ " " + str(x2) + ", " + str(y1) + " " + str(y2))
	
	binary_image = tmp_img
	tmp_img_re = tmp_img[x1 : x2, y1: y2]

	#トリミングした画素を16区画に分ける=sepa1~sepa16に画像のデータが入る
	separate(tmp_img_re)

	count = 1
	point_dict = {} #point_dictを初期化
	for d in list_sepa:
		if (count == 6 or count == 7 or count == 10 or count ==11):
			point = review(d, 1)
			point_dict[count] = point
			print(str(count)+ " : " + str(point))
			count += 1

		else:
			point = review(d, 0.2)
			point_dict[count] = point
			print(str(count)+ " : " + str(point))
			count += 1

	#pointと画素の番号を辞書型で保存
	sortedDict = sorted(point_dict.items(), key=lambda x:x[1], reverse=True)#list型
	#print(point_dict)

	#cv.imshow("Most", list_sepa[sortedDict[0][0] - 1])
	#cv.imshow("Second", list_sepa[sortedDict[1][0] - 1])


	#元画像での区画中心の座標を計算
	#ここの時点で一区画の大きさがwidthとheightに入っている
	#print(width)
	#print(height)
	#print(str(sortedDict[0][0]) + ", " + str(sortedDict[1][0]))
	tmp_x1 = x1 + int(height / 2) + int(sortedDict[0][0] % 4.5) * height#Maxの画像の中心座標ｘ
	tmp_y1 = y1 + int(width / 2) + int(int(sortedDict[0][0]) / 5) * width#Maxの画像の中心座標ｙ
	point1 = np.array([tmp_x1, tmp_y1])#Maxの画像の中心座標
	tmp_x2 = x1 + int(height / 2) + int(sortedDict[1][0] % 4.5) * height#Secondの画像の中心座標ｘ
	tmp_y2 = y1 + int(width / 2) + int(int(sortedDict[1][0]) / 5) * width#Secondの画像の中心座標ｙ
	point2 = np.array([tmp_x2, tmp_y2])#Secondの画像の中心座標
	print(str(tmp_x1) + ", " + str(tmp_y1) + " " + str(tmp_x2) + ", " + str(tmp_y2))

	#傷の重心を計算(黒だったところの多さで内分点を決定する)
	#p1 = int(sortedDict[0][1] / N)
	#p2 = int(sortedDict[1][1] / N)
	#tmp_x = int((point1[0] * p1 + point2[0] * p2) / (p1 + p2))
	#tmp_y = int((point1[1] * p1 + point2[1] * p2) / (p1 + p2))

	#中点を重心にする
	tmp_x = int((point1[0] + point2[0]) / 2)
	tmp_y = int((point1[1] + point2[1]) / 2)
	g_point = np.array([tmp_x, tmp_y])#傷の重心
	#print(g_point)


def detect_figure(img):#重心を使って最短辺から最長辺を求める
	find_gravity(img)
	find_gravity_b(img)
	global tmp_img, g_point, min_point1,min_point2, max_point1, max_point2, tmp_point1, tmp_point2
	global short_axi, long_axi#最短辺,最長辺
	global x , y, k, b#係数と変数
	global binary_image#2値画像
	tmp_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#グレースケールに変換
	tmp_img = cv.GaussianBlur(tmp_img, (5, 5), 3)#ガウシアンフィルタ
	tmp_img = cv.GaussianBlur(tmp_img, (5, 5), 3)#ガウシアンフィルタ
	tmp_img = cv.bilateralFilter(tmp_img, 15, 20, 20)#バイラテラルフィルタをかける
	tmp_img = cv.GaussianBlur(tmp_img, (5, 5), 3)#ガウシアンフィルタ
	tmp_img = cv.Canny(tmp_img, 50, 110)#エッジ検出
	min = N
	tmp_point = []
	dir = 0

	#Harrisのコーナー検出(使ってない)
	gray = np.float32(tmp_img)
	dst = cv.cornerHarris(gray, 2, 3, 0.01)
	dst = cv.dilate(dst, None)
	#print(dst)
	#img[dst>0.01*dst.max()]=[255, 0,0]


	#最短辺を求める
	#重心から最も近いエッジを検出
	short_axi = 1000
	long_axi = 0
	tmp_point1 = []
	tmp_point2 = []
	
	for r in range(179):#tanの発散を防ぐために数値設定
		k = round(math.tan(math.radians(r - 89)),3)
		b = round(g_point[1] - k * g_point[0], 3)
		for i in range(tmp_img.shape[0]- g_point[0]):
				x = g_point[0] + i
				y = int(round(k * x + b))
				if(y < 0 or tmp_img.shape[1] - 2 < y):#幅３なのでその分ｙの範囲が狭まる
					continue
			
				elif(tmp_img[x][y].tolist() == 255 or tmp_img[x][y + 1].tolist() == 255 or tmp_img[x][y - 1].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point1 = np.array([y, x])
					break
				
		for i in range(g_point[0]):
				x = g_point[0] - i
				y = int(round(k * x + b))
				
				if(y < 0 or tmp_img.shape[1] - 2 <= y):
					continue
					
				elif(tmp_img[x][y].tolist() == 255 or tmp_img[x][y + 1].tolist() == 255 or tmp_img[x][y - 1].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point2 = np.array([y, x])#ベクトルを保存
					break
		
		if(tmp_point1 == [] or tmp_point2 == []):
			continue
		
		#min_point1と２に最短辺の座標が入ってる
		#最短辺の長さを算出
		#print(str(tmp_point1) +" , "+ str(tmp_point2))
		tmp_point1 = np.array(tmp_point1)
		tmp_point2 = np.array(tmp_point2)
			
		if(long_axi < round(np.linalg.norm(tmp_point1 - tmp_point2))):
			max_point1 = tmp_point1
			max_point2 = tmp_point2
			long_axi = round(np.linalg.norm(tmp_point1 - tmp_point2))
			print("long : "+ str(long_axi))
		
		tmp_point1 = []
		tmp_point2 = []

if __name__ == '__main__':
		img = cv.imread(path)
		#detect_figure(img)
		g_point = find_gravity_r(img)

		#cv.drawMarker(img, (point1[0], point1[1]), (255, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (point2[0], point2[1]), (255, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		cv.drawMarker(img, (g_point[0], g_point[1]), (0, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (min_point1[0], min_point1[1]), (0, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (min_point2[0], min_point2[1]), (0, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (max_point1[0], max_point1[1]), (255, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (max_point2[0], max_point2[1]), (255, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		cv.imshow("incision1",img)
		#cv.imshow("incision2",tmp_img)
		#cv.imshow("incision3",binary_image)
		cv.waitKey()
