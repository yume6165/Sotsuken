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

def detect_figure(img):#重心を使って最短辺から最長辺を求める
	
	global binary_image#2値画像
	
	g_point = find_gravity_r(img)
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
	#cv.drawMarker(img, (g_point[0], g_point[1]), (255, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)	
	#cv.imshow("img", tmp_img)
	#最短辺を求める
	#重心から最も近いエッジを検出
	for i in range(tmp_img.shape[0]):#多分縦方向
		for j in range(tmp_img.shape[1]):#多分横方向
			#print(tmp_img[dst>0.01*dst.max()])
			#print(tmp_img[i][j].tolist())
			if(tmp_img[i][j].tolist() == 255):#エッジ（白）ならば
				tmp_point = np.array([j, i])#ベクトルを保存

				dir = np.linalg.norm(g_point - tmp_point)
				if(dir < min):
					min = dir
					min_point1 = tmp_point
					
	
	#重心と一番近いエッジのｘ座標が等しいとき
	if(g_point[0] == min_point1[0]):
		x = g_point[0]
		
		cv.drawMarker(img, (min_point1[0], min_point1[1]), (0, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
		#cv.imshow("min_point",img)
		
		if(min_point1[1] < g_point[1]):#最短点からみて重心の反対側を探す
			for i in range(tmp_img.shape[0]):
				y = g_point[1] + i
			
				if(y < 0 or y >= tmp_img.shape[0] - 1):
					contnue
				elif(tmp_img[y][x].tolist() == 255 or tmp_img[y - 1][x].tolist() == 255 or tmp_img[y + 1][x].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point = np.array([x, y])#ベクトルを保存
					min_point2 = tmp_point
					break
		else:
			for i in range(tmp_img.shape[0]):
				y = g_point[1] - i
			
				if(y < 0 or y >= tmp_img.shape[0]-1):
					contnue
				elif(tmp_img[y][x].tolist() == 255 or tmp_img[y + 1][x].tolist() == 255 or tmp_img[y - 1][x].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point = np.array([x, y])#ベクトルを保存
					min_point2 = tmp_point
					break
		#print(min_point2)
		#cv.drawMarker(img, (min_point2[0], min_point2[1]), (0, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.imshow("min_point",img)
		
	#最短点から傷の幅を計算y=kx + b
	else:
		k = (g_point[1] - min_point1[1]) / (g_point[0] - min_point1[0])
		b = g_point[1] - k * g_point[0]
		print("k is "+str(k)+" , b is"+str(b))
		#print(min_point1)
		#print(g_point)

		#cv.drawMarker(img, (min_point1[0], min_point1[1]), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
		#cv.imshow("min_point",img)		
	
		if(min_point1[0] < g_point[0]):#最短点からみて重心の反対側を探す
			for i in range(tmp_img.shape[0]):
				x = g_point[0] + i
				y = int(k * x + b)
			
				if(x >= tmp_img.shape[1] + 1):
					continue
				elif(y < 0 or y >= tmp_img.shape[0] + 2):
					contnue
				elif(tmp_img[y][x].tolist() == 255 or tmp_img[y][x + 1].tolist() == 255 or tmp_img[y][x - 1].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point = np.array([x, y])#ベクトルを保存
					min_point2 = tmp_point
					break
		else:
			for i in range(tmp_img.shape[0]):
				x = g_point[0] - i
				y = int(k * x + b)
			
				if(x >= tmp_img.shape[1] + 1):
					continue
				elif(y < 0 or y >= tmp_img.shape[0] + 2):
					contnue
				elif(tmp_img[y][x].tolist() == 255 or tmp_img[y][x + 1].tolist() == 255 or tmp_img[y][x - 1].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point = np.array([x, y])#ベクトルを保存
					min_point2 = tmp_point
					break
	
	#min_point1と２に最短辺の座標が入ってる
	#最短辺の長さを算出
	if(len(min_point2) == 0):
		short_axi = round(np.linalg.norm(min_point1 - g_point))*2
	elif(len(min_point1) == 0):
		short_axi = round(np.linalg.norm(min_point2 - g_point))*2
	elif(len(min_point2) != 0 and len(min_point1) != 0):
		short_axi = round(np.linalg.norm(min_point1 - min_point2))
	else:
		print("MIN_POINT Error")

	#最長辺を計算
	long_axi = 0
	tmp_point1 = g_point
	tmp_point2 = g_point
	
	for r in range(179):#tanの発散を防ぐために数値設定
		k = round(math.tan(math.radians(r - 89)),3)
		b = round(g_point[1] - k * g_point[0], 3)
		for i in range(tmp_img.shape[0]- g_point[1]):
				x = g_point[0] + i
				y = int(round(k * x + b))
				if(y < 0 or tmp_img.shape[0] - 2 < y):#幅３なのでその分ｙの範囲が狭まる
					continue
			
				elif(tmp_img[y][x].tolist() == 255 or tmp_img[y + 1][x].tolist() == 255 or tmp_img[y - 1][x].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point1 = np.array([x, y])
					break
				
		for i in range(g_point[1]):
				x = g_point[0] - i
				y = int(round(k * x + b))
				#print(str(x) +", "+str(y))
	
				if(y < 0 or tmp_img.shape[0] - 2 <= y):
					continue
					
				elif(tmp_img[y][x].tolist() == 255 or tmp_img[y + 1][x].tolist() == 255 or tmp_img[y - 1][x].tolist() == 255):#エッジ（白）ならば,幅は３
					tmp_point2 = np.array([x, y])#ベクトルを保存
					break
		
		if(len(tmp_point1) == 0 or len(tmp_point2) == 0):
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
		
	#cv.drawMarker(img, (max_point1[0], max_point1[1]), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
	#cv.drawMarker(img, (max_point2[0], max_point2[1]), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
	#cv.imshow("points",img)
	
	return max_point1, max_point2, min_point1, min_point2, long_axi, short_axi
	
def detect_edge(img):#形を求める
	tmp_img = edge_detection(img)
	max_point1, max_point2, min_point1, min_point2, long_axi, short_axi = detect_figure(img)
	
	g_point = find_gravity_r(img)
	k = (max_point1[1] - max_point2[1])/(max_point1[0] - max_point2[0])
	b = max_point1[1] - k * max_point1[0]
	size = []
	distance = []
	sin_x = []
	sin_y = []
	print(max_point1)
	print(max_point2)
	
	if(abs(k) < 1):#傾きが小さく殆ど水平の時　x= g_point[0]）
		for i in range(int(long_axi)):
			x = max_point1[0] - i
			y = int(round(k * x + b))
			c = y + 1 / k * x
			tmp_point1 = []
			tmp_point2 = []
			
			for j in range(int(round(long_axi))):
				x1 = x
				y1 = y + j
				print(str(x1)+", "+str(y1))
				#cv.drawMarker(img_edge, (x1, y1), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				
				
				if(y1 < 1 or tmp_img.shape[0] - 1 <= y1):
					continue
					
				elif(x1 < 1 or tmp_img.shape[1] - 1 <= x1):
					continue
				
				else:
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1 + 1][x1].tolist() == 255 or tmp_img[y1 - 1][x1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						break
				
			for j in range(int(round(long_axi))):
				x1 = x
				y1 = y - j
				#cv.drawMarker(img_edge, (x1, y1), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				if(y1 < 1 or tmp_img.shape[0] - 1 <= y1):
					continue
					
				elif(x1 < 1 or tmp_img.shape[1] - 1 <= x1):
					continue
				
				else:
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1 + 1][x1].tolist() == 255 or tmp_img[y1 - 1][x1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point2 = np.array([x1, y1])#ベクトルを保存
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						break
					
			if(tmp_point1 != [] and tmp_point2 != []):
				print(round(np.linalg.norm(tmp_point1 - tmp_point2)))
				cv.drawMarker(img_edge, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, (tmp_point1[0], tmp_point1[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, (tmp_point2[0], tmp_point2[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, tuple(max_point1), (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, tuple(max_point2), (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				
				size.append(round(np.linalg.norm(tmp_point1 - tmp_point2), 3))
				distance.append(i)
				
			elif(tmp_point1 == [] or tmp_point2 == []):
				print("Skip")
				continue
		
		
	
	elif(1 < abs(k) and max_point1[0] > max_point2[0]):
		for i in range(int(long_axi)):
			x = max_point1[0] - i
			y = int(round(k * x + b))
			c = y + 1 / k * x
			tmp_point1 = []
			tmp_point2 = []
			
			for j in range(int(round(long_axi))):
				x1 = int(x + j)
				y1 = int(round(-1 / k * x1 + c))
				print(str(x1)+", "+str(y1))
				cv.drawMarker(img_edge, (x1, y1), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				
				
				if(y1 < 1 or tmp_img.shape[0] - 1 <= y1):
					continue
					
				elif(x1 < 1 or tmp_img.shape[1] - 1 <= x1):
					continue
				
				else:
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1 + 1][x1].tolist() == 255 or tmp_img[y1 - 1][x1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						break
				
			for j in range(int(round(long_axi))):
				x1 = int(x - j)
				y1 = int(round(-1 / k * x1 + c))
				cv.drawMarker(img_edge, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				if(y1 < 1 or tmp_img.shape[0] - 1 <= y1):
					continue
					
				elif(x1 < 1 or tmp_img.shape[1] - 1 <= x1):
					continue
				
				else:
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1 + 1][x1].tolist() == 255 or tmp_img[y1 - 1][x1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point2 = np.array([x1, y1])#ベクトルを保存
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						break
					
			if(tmp_point1 != [] and tmp_point2 != []):
				print(round(np.linalg.norm(tmp_point1 - tmp_point2)))
				cv.drawMarker(img_edge, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, (tmp_point1[0], tmp_point1[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, (tmp_point2[0], tmp_point2[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, tuple(max_point1), (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, tuple(max_point2), (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				
				size.append(round(np.linalg.norm(tmp_point1 - tmp_point2), 3))
				distance.append(i)
				
			elif(tmp_point1 == [] or tmp_point2 == []):
				print("Skip")
				continue
			
			
	elif(1 < abs(k) and max_point1[0] <= max_point2[0]):
		for i in range(int(round(long_axi))):		
			x = max_point1[0] + i
			y = int(round(k * x + b))
			c = y + 1 / k * x
			tmp_point1 = []
			tmp_point2 = []
			cv.drawMarker(img, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
			
			for j in range(int(round(short_axi * 1.5))):
				x1 = int(x + j)
				y1 = int(round(-1 / k * x1 + c))
				
				if(y1 < 1 or tmp_img.shape[0] - 1<= y1):
					continue
					
				elif(x1 < 1 or tmp_img.shape[1] - 1 <= x1):
					continue
				
				else:
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1 + 1][x1].tolist() == 255 or tmp_img[y1 - 1][x1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						break
				
			for j in range(int(round(short_axi * 1.5))):
				x1 = int(x - j)
				y1 = int(round(-1 / k * x1 + c))
				
				if(y1 < 1 or tmp_img.shape[0] - 1 <= y1):
					continue
					
				elif(x1 < 1 or tmp_img.shape[1] - 1 <= x1):
					continue
				
				else:
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1 + 1][x1].tolist() == 255 or tmp_img[y1 - 1][x1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point2 = np.array([x1, y1])#ベクトルを保存
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						break
				
				
			#print(str(tmp_point1) + " , "+ str(tmp_point2))
			
			if(tmp_point1 != [] and tmp_point2 != []):
				print(round(np.linalg.norm(tmp_point1 - tmp_point2)))
				cv.drawMarker(img_edge, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, max_point1, (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, max_point2, (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, (tmp_point1[0], tmp_point1[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				cv.drawMarker(img_edge, (tmp_point2[0], tmp_point2[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				size.append(round(np.linalg.norm(tmp_point1 - tmp_point2), 3))
				distance.append(i)
				
			elif(tmp_point1 == [] or tmp_point2 == []):
				print("Skip")
				continue
			
				
	#print(size)
	cv.drawMarker(img_edge, (g_point[0], g_point[1]), (0, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
	cv.imshow("edge",img_edge)
	cv.imshow("img",tmp_img)
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
			
	if(len(edge_point) == 0):
		return "Error"
			
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
	
	#plt.show()
	
	if(num < 0):#上に凸なら
		return "pull"
		
	else:
		return "push"
	
	#cv.imshow("tmp_img",tmp_img)
	#cv.imshow("img",img)
	
def judge(img):
	sharp = False
	oval = False
	pull = False
	push = False
	
	detect_figure(img)
	detect_edge(img)
	
	#s = pullpush_judge(img)	
	#if(s == "pull"):
	#	pull = True
	#	push = False
	#elif(s == "push"):
	#	pull = False
	#	push = True
	#else:
	#	pull = False
	#	push = False
		
	#oval = oval_judge(img)
	#sharp = sharp_judge(img)
	
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
		#cv.drawMarker(img, (g_point[0], g_point[1]), (0, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (min_point1[0], min_point1[1]), (0, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (min_point2[0], min_point2[1]), (0, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (max_point1[0], max_point1[1]), (255, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.drawMarker(img, (max_point2[0], max_point2[1]), (255, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
		#cv.imshow("incision1",img)
		#cv.imshow("incision2",tmp_img)
		#cv.imshow("incision3",binary_image)
		cv.waitKey()
