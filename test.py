#codeing:utf-8
# -*- coding: utf-8 -*-

import os, sys, time
import cv2 as cv
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

#研究室で研究するとき
#path = "./sample/incision_1.jpg"

#ノートパソコンで研究するとき
path = "D:\Sotsuken\Sotsuken_repo./sample/incision_1.jpg"

N = 1000

def edge_detection(img):
	tmp_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#グレースケールに変換
	tmp_img = cv.GaussianBlur(tmp_img, (5, 5), 3)#ガウシアンフィルタ
	tmp_img = cv.GaussianBlur(tmp_img, (5, 5), 3)#ガウシアンフィルタ
	tmp_img = cv.bilateralFilter(tmp_img, 15, 20, 20)#バイラテラルフィルタをかける
	tmp_img = cv.GaussianBlur(tmp_img, (5, 5), 3)#ガウシアンフィルタ
	tmp_img = cv.Canny(tmp_img, 50, 110)#エッジ検出
	return tmp_img

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
	#	cv.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), (0, 0, 255), thickness=2)
	#	cv.imshow('red', img)
		
	#最大の四角を見つける
	result_rect = max(rects, key=(lambda x: x[2] * x[3]))
			
	#cv.rectangle(img, tuple(result_rect[0:2]), tuple(result_rect[0:2] + result_rect[2:4]), (0, 255, 0), thickness=2)
	#print(result_rect)
	re_img = img[ result_rect[1] : result_rect[1] + result_rect[3], result_rect[0] : result_rect[0] + result_rect[2]]
	x = result_rect[0] + int(round(result_rect[2]/2))
	y = result_rect[1] + int(round(result_rect[3]/2))
	#cv.imshow('re_red', re_img)
	g_point = np.array([x, y])
	
	return g_point


def find_gravity(img):#傷の重心を探す関数
	global tmp_img, width, height, x1, x2, y1, y2, N#画像処理のための一時的な保管場所
	global tmp_img_re, list_sepa, binary_image#２値画像
	global point_dict, point1, point2, g_point, tmp_x, tmp_y

	#パラメータ
	width = img.shape[0]
	height = img.shape[1]
	threshold = 110
	cut_rate = 70 #トリミングするときの比率
	comp_rate = 25 #4*4で画像を区切って

	#print(str(width) + " " + str(height))


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
			#print(str(count)+ " : " + str(point))
			count += 1

		else:
			point = review(d, 0.2)
			point_dict[count] = point
			#print(str(count)+ " : " + str(point))
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
	#print(str(tmp_x1) + ", " + str(tmp_y1) + " " + str(tmp_x2) + ", " + str(tmp_y2))

	#傷の重心を計算(黒だったところの多さで内分点を決定する)
	#p1 = int(sortedDict[0][1] / N)
	#p2 = int(sortedDict[1][1] / N)
	#tmp_x = int((point1[0] * p1 + point2[0] * p2) / (p1 + p2))
	#tmp_y = int((point1[1] * p1 + point2[1] * p2) / (p1 + p2))

	#中点を重心にする
	tmp_x = int((point1[0] + point2[0]) / 2)
	tmp_y = int((point1[1] + point2[1]) / 2)
	g_point = np.array([tmp_x, tmp_y])#傷の重心
	#print("gravity:"+str(g_point))
	return g_point


def detect_figure(img):#重心を使って最短辺から最長辺を求める
	global tmp_img, g_point, min_point1,min_point2, max_point1, max_point2, tmp_point
	global short_axi, long_axi#最短辺
	global x , y, k, b#係数と変数
	global binary_image#2値画像
	tmp_img = edge_detection(img)
	min = N
	tmp_point = []
	min_point = []
	min_point1 = []
	min_point2 = []
	max_point1 = []
	max_point2 = []
	dir = 0

	#Harrisのコーナー検出(使ってない)
	gray = np.float32(tmp_img)
	dst = cv.cornerHarris(gray, 2, 3, 0.01)
	dst = cv.dilate(dst, None)
	#print(dst)
	#img[dst>0.01*dst.max()]=[255, 0,0]


	#最短辺を求める
	#重心から最も近いエッジを検出
	for i in range(len(dst)):#多分縦方向
		for j in range(len(dst[0])):#多分横方向
			#print(tmp_img[dst>0.01*dst.max()])
			#print(tmp_img[i][j].tolist())
			if(tmp_img[i][j].tolist() == 255):#エッジ（白）ならば
				tmp_point = np.array([i, j])#ベクトルを保存

				dir = np.linalg.norm(g_point - tmp_point)
				if(dir < min):
					min = dir
					min_point1 = tmp_point
					
	
	#最短点から傷の幅を計算y=kx + b
	k = (g_point[1] - min_point1[1]) / (g_point[0] - min_point1[0])
	b = g_point[1] - k * g_point[0]
	#print("k is "+str(k)+" , b is"+str(b))
	
	if(min_point1[0] <= g_point[0]):#最短点からみて重心の反対側を探す
		for i in range(tmp_img.shape[0]):
			y = g_point[0] + i
			x = int(k * y + b)
			
			if(x >= tmp_img.shape[0] + 1):
				continue
			elif(y < 0 or y >= tmp_img.shape[1] + 2):
				contnue
			elif(tmp_img[x][y].tolist() == 255 or tmp_img[x][y + 1].tolist() == 255 or tmp_img[x][y - 1].tolist() == 255):#エッジ（白）ならば,幅は３
				tmp_point = np.array([x, y])#ベクトルを保存
				min_point2 = tmp_point
				break
	else:
		for i in range(tmp_img.shape[0]):
			x = g_point[0] - i
			y = int(k * x + b)
			
			if(x >= tmp_img.shape[0] + 1):
				continue
			elif(y < 0 or y >= tmp_img.shape[1] + 2):
				contnue
			elif(tmp_img[x][y].tolist() == 255 or tmp_img[x][y + 1].tolist() == 255 or tmp_img[x][y - 1].tolist() == 255):#エッジ（白）ならば,幅は３
				tmp_point = np.array([x, y])#ベクトルを保存
				min_point2 = tmp_point
				break
	
	#min_point1と２に最短辺の座標が入ってる
	#最短辺の長さを算出
	if(min_point2 == []):
		short_axi = round(np.linalg.norm(min_point1 - g_point))
	elif(min_point1 == []):
		short_axi = round(np.linalg.norm(min_point2 - g_point))
	else:
		short_axi = round(np.linalg.norm(min_point1 - min_point2))
	#print(short_axi)
	

	#最長辺を計算
	long_axi = 0
	tmp_point1 = g_point
	tmp_point2 = g_point
	
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
		#print(str(tmp_point1) +" , "+ str(tmp_point2))
		tmp_point1 = np.array(tmp_point1)
		tmp_point2 = np.array(tmp_point2)
			
		if(long_axi < round(np.linalg.norm(tmp_point1 - tmp_point2))):
			max_point1 = tmp_point1
			max_point2 = tmp_point2
			long_axi = round(np.linalg.norm(tmp_point1 - tmp_point2))
			#print("long : "+ str(long_axi))
		
		tmp_point1 = []
		tmp_point2 = []
	
	return max_point1, max_point2, min_point1, min_point2, long_axi, short_axi
	
def detect_edge(img):#形を求める
	global tmp_point1, tmp_point2, x, y, img_edge
	tmp_img = edge_detection(img)
	max_point1, max_point2, min_point1, min_point2, long_axi, short_axi = detect_figure(img)
	
	k = (max_point1[1] - max_point2[1])/(max_point1[0] - max_point2[0])
	b = max_point1[1] - k * max_point1[0]
	size = []
	distance = []
	sin_x = []
	sin_y = []

	
	if(max_point1[0] > max_point2[0]):
		
		for i in range(int(long_axi)):
			x = max_point1[0] - i
			y = int(round(k * x + b))
			c = y + 1 / k * x
			tmp_point1 = []
			tmp_point2 = []
			
			for j in range(int(round(long_axi))):
				x1 = int(x + j)
				y1 = int(round(-1 / k * x1 + c))
				
				if(y1 < 0 or tmp_img.shape[1] - 2 <= y1):
					continue
					
				elif(x1 < 0 or tmp_img.shape[0] <= x1):
					continue
				
				elif(tmp_img[x1][y1].tolist() == 255 or tmp_img[x1][y1 + 1].tolist() == 255 or tmp_img[x1][y1 - 1].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point1 = np.array([y1, x1])#ベクトルを保存
					break
				
			for j in range(int(round(long_axi))):
				x1 = int(x - j)
				y1 = int(round(-1 / k * x1 + c))
				
				if(y1 < 0 or tmp_img.shape[1] - 2 <= y1):
					continue
					
				elif(x1 < 0 or tmp_img.shape[0] <= x1):
					continue
				
				elif(tmp_img[x1][y1].tolist() == 255 or tmp_img[x1][y1 + 1].tolist() == 255 or tmp_img[x1][y1 - 1].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point2 = np.array([y1, x1])#ベクトルを保存
					break
					
			if(tmp_point1 != [] and tmp_point2 != []):
				print(round(np.linalg.norm(tmp_point1 - tmp_point2)))
			
	elif(max_point1[0] <= max_point2[0]):
		for i in range(int(round(long_axi))):		
			x = max_point1[0] + i
			y = int(round(k * x + b))
			c = y + 1 / k * x
			tmp_point1 = []
			tmp_point2 = []
			#cv.drawMarker(img, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				
			
			
			for j in range(int(round(short_axi * 1.5))):
				x1 = int(x + j)
				y1 = int(round(-1 / k * x1 + c))
				
				if(y1 < 0 or tmp_img.shape[1] - 2 <= y1):
					continue
					
				elif(x1 < 0 or tmp_img.shape[0] <= x1):
					continue
				
				elif(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 - 1].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point1 = np.array([y1, x1])#ベクトルを保存
					break
				
			for j in range(int(round(short_axi * 1.5))):
				x1 = int(x - j)
				y1 = int(round(-1 / k * x1 + c))
				
				if(y1 < 0 or tmp_img.shape[1] - 2 <= y1):
					continue
					
				elif(x1 < 0 or tmp_img.shape[0] <= x1):
					continue
				
				elif(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 - 1].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point2 = np.array([y1, x1])#ベクトルを保存
					break
				
				
			#print(str(tmp_point1) + " , "+ str(tmp_point2))
			
			if(tmp_point1 == [] or tmp_point2 == []):
				#print("Skip")
				continue
			
			else:
				cv.drawMarker(img_edge, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, (tmp_point1[1], tmp_point1[0]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, (tmp_point2[1], tmp_point2[0]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				size.append(round(np.linalg.norm(tmp_point1 - tmp_point2), 3))
				distance.append(i)
				
	#print(size)
	cv.imshow("edge",img_edge)
	
	return size, distance, short_axi
	

def oval_judge(img):
	size, distance, short_axi = detect_edge(img)
	
	#傷が円に近いほどsinとの誤差が小さくなる
	sin_x = np.arange(0,distance.index(max(distance))+1, 1)
	sin_y = np.sin(sin_x/(distance.index(max(distance)))*math.pi)*short_axi
	
	#差分は正規化
	difference = abs((size - sin_y)*(size - sin_y))/short_axi/short_axi
	#plt.scatter(distance, size, label="difference", color="red")
	#plt.plot(sin_x, sin_y, label="difference", color="green")
	#plt.show()
	difference = sum(difference)/len(difference)#平均
	print("Oval:"+str(difference))
	
	if (difference <= 0.02):
		return True
	else:
		return False
		

def sharp_judge(img):
	size, distance, short_axi = detect_edge(img)
	cos_x = np.arange(0,distance.index(max(distance))+1, 1)
	cos_y = np.cos(cos_x/(distance.index(max(distance)))*math.pi)*short_axi
	
	#傷の幅を3分割にする
	sp_size = list(np.array_split(size, 3))
	sp_distance = list(np.array_split(distance, 3))
	sp_cos_x = list(np.array_split(cos_x, 3))
	sp_cos_y = list(np.array_split(cos_y, 3))
	
	#差分を計算(fw:前)
	fw_size = np.diff(sp_size[0], n=1)
	fw_distance = [x for x in range(len(fw_size))]
	fw_cos_x = [x for x in range(len(fw_size))]
	fw_cos_y = np.cos(np.array(fw_cos_x)/(distance.index(max(distance)))*math.pi)*short_axi
	
	#円の増加速度と比較
	fw_result = fw_size / fw_cos_y
	fw_result = round(max(fw_result), 3)
	
	#差分を計算(bw:後)
	bw_size = np.diff(sp_size[2], n=1)
	bw_distance = [x for x in range(len(sp_distance[0])+len(sp_distance[1]), len(distance)-1)]
	bw_cos_x = [x for x in range(len(sp_distance[0])+len(sp_distance[1]), len(distance)-1)]
	bw_cos_y = np.cos(np.array(bw_cos_x)/(distance.index(max(distance)))*math.pi)*short_axi
	
	#円の増加速度と比較
	bw_result = bw_size / bw_cos_y
	bw_result = round(max(bw_result), 3)
	
	
	#サイン関数とプロット、近似曲線を表示
	#plt.scatter(distance, size, label="sharp", color="red")	
	#plt.plot(cos_x, cos_y, label="cos", color="green")
	#plt.plot(distance, np.poly1d(np.polyfit(distance, size, 2))(distance), label="近似", color="red")
	#cv.imshow("img",img)
	plt.show()
	
	if((fw_result + bw_result) / 2 < 0.5):#0.5未満ならシャープ
		return True
		
	else:
		return False
		

def contrast(image, a):#(aはゲイン)
	image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)#グレースケールに変換
	lut = [ np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.))) for i in range(256)] 
	result_image = np.array( [ lut[value] for value in image.flat], dtype=np.uint8 )
	result_image = result_image.reshape(image.shape)
	return result_image


def pullpush_judge(img):#文字列でpullかpushかを返します
	global light_point, kaizoudo
	gravity = find_gravity(img)#重心
	img_list = []#9区画に分けた画像を保存
	img_ave = []
	result_list = []
	
	#照明の方向を検出
	tmp_img = contrast(img, 10)#コントラストを上げる
	width = tmp_img.shape[0]
	height = tmp_img.shape[1]
	w = int(round(width/3))
	h = int(round(height/3))
	
	for i in range(3):#9区画に分けます
		for j in range(3):
			g = tmp_img[j * w : (j + 1) * w, i * h : (i + 1) * h]
			img_list.append(g)
	
	#print(img_list)
	
	#平均の明るさを求める
	for i in img_list:
		ave = 0
		for j in range(i.shape[0]):
			ave += sum(i[j]) / len(i[j])
			
		img_ave.append(round(ave / i.shape[0], 3))
	
	#最も明るいマスを判定
	num = img_ave.index(max(img_ave))
	cv.imshow("ligth", img_list[num])
	
	#最も明るいマスの中心と重心を結んだ直線を計算
	if(num <= 2):
		x = int(round(h / 2))
		y = int(round(w * num % 3 + w / 2))
	elif(num >= 6):
		x = int(round(h / 2 + 2 * h))
		y = int(round(w * num % 3 + w / 2))
	else:
		x = int(round(h / 2 + h))
		y = int(round(w * num % 3 + w / 2))
	
	light_point = np.array([x, y])
	#print(light_point)
	#cv.drawMarker(tmp_img, (x, y), (0, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
	k = (gravity[1]/ light_point[1]) /(gravity[0]/ light_point[0]) 
	b = gravity[1] - k * gravity[0]
	
	edge_img = edge_detection(img)
	edge_point = []
	
	for i in range(height):
		y = int(gravity[0] - (gravity[0] - light_point[0])/abs(gravity[0] - light_point[0])*i)
		x = int(round(k * y + b))
		
		if(x < 0 or y < 0 or edge_img.shape[0] < x or edge_img.shape[1] < y):
			continue
		
		#cv.drawMarker(tmp_img, (x, y), (30, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)

		if(edge_img[x][y].tolist() == 255 or edge_img[x+1][y].tolist() == 255 or edge_img[x-1][y].tolist() == 255):
			print(str(x)+" , "+str(y))
			cv.drawMarker(tmp_img, (x, y), (0, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
			edge_point = np.array([y, x])
			break
			
	for i in range(int(10*img.shape[0]*img.shape[1]/40000)):
		x1 = edge_point[0] + i
		y1 = int(round(k * x1 + b))
		x2 = edge_point[0] + i
		y2 = int(round(k * x2 + b + 1))
		x3 = edge_point[0] + i
		y3 = int(round(k * x3 + b - 1))
		
		result = round((tmp_img[y1][x1] + tmp_img[y2][x2] + tmp_img[y3][x3])/3, 3)
		result_list.append(result)
	
	result_list_x = [x for x in range(len(result_list))]
	num = np.polyfit(result_list_x, result_list, 2)[0]
	plt.scatter(result_list_x, result_list, label="light", color="red")
	plt.plot(result_list_x, np.poly1d(np.polyfit(result_list_x, result_list, 2))(result_list_x), label="近似", color="red")
	
	plt.show()
	
	if(num < 0):#上に凸なら
		return "pull"
		
	else:
		return "push"
	
	#cv.imshow("tmp_img",tmp_img)
	#cv.imshow("img",img)
	
def judge(img):
	gravity = find_gravity(img)
	detect_figure(img)
	
	s = pullpush_judge(img)	
	if(s == "pull"):
		pull = True
		push = False
	else:
		pull = False
		push = True
		
	oval = oval_judge(img)
	sharp = sharp_judge(img)
	
	return sharp, oval, pull, push

if __name__ == '__main__':
		img = cv.imread(path)
		img_edge = cv.imread(path)
		sharp, oval, pull, push = judge(img)
		
		print("鋭さ："+str(sharp)+"　円度："+ str(oval)+"　引き："+ str(pull)+"　押し："+ str(push))
		
		#gravity = find_gravity(img)
		#detect_figure(img)
		
		#以下は確認用のマーカ
		#cv.drawMarker(img, (point1[0], point1[1]), (255, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (point2[0], point2[1]), (255, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		cv.drawMarker(img, (g_point[0], g_point[1]), (0, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (min_point1[0], min_point1[1]), (0, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (min_point2[0], min_point2[1]), (0, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (max_point1[0], max_point1[1]), (255, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (max_point2[0], max_point2[1]), (255, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		cv.imshow("incision1",img)
		cv.imshow("incision2",tmp_img)
		cv.imshow("incision3",binary_image)
		cv.waitKey()
