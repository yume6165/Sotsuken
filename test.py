#codeing:utf-8
# -*- coding: utf-8 -*-

import os, sys, time
import cv2 as cv
from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import math
import cmath
import matplotlib.pyplot as plt
import glob
from statistics import mean, stdev
import seaborn as sns
import csv
import collections as cl

#研究室で研究するとき
#path = "./sample/incision_1.jpg"

#ノートパソコンで研究するとき
path = "D:\Sotsuken\Sotsuken_repo\sample\\*"

N = 1000

thresh = 1.0E-10

#
#
#意味の数学モデル
#
#

#閾値
e = -1


def flatten(data):
    for item in data:
        if hasattr(item, '__iter__'):
            for element in flatten(item):
                yield element
        else:
            yield item
			
			
def no_context_dist(data):#なんの文脈もないときの距離を計算
	distances = []
	for d1 in data:
		dist = []
		for d2 in data:
			tmp = np.array(d1) - np.array(d2)
			dst = np.linalg.norm(tmp)
			dist.append(dst)
		distances.append(dist)
	
	#print(distances)
	return distances

def make_semantic_matrix(data_mat):#意味行列を作るためのに作成したセマンティックな行列を入力
	
	#相関行列の作成
	relation_mat = np.dot(np.array(data_mat).T, np.array(data_mat))
	eig_val, eig_vec = np.linalg.eig(relation_mat)#固有値と固有ベクトルを取得
	#print(eig_val)
	#print(eig_vec)
	
	U,_ = np.linalg.qr(eig_vec)
	
	#print(len(U))
	
	#result_sem_mat =[]
	#for vec in eig_vec:
	#	if(np.linalg.norm(vec) > 0):
	#		#print(vec)
	#		result_sem_mat.append(vec)
	
	#print("sem")
	#print(len(result_sem_mat))
	
	return U
	

#文脈の作成
def make_context(sem_mat, word_list, contex_word, data):#意味行列に文脈を指定する単語リストを入力
	global e
	contex_mat = []
	count = 0
	
	#相関行列(データ行列では列が画像行が単語なので代わりに相関行列を利用)
	relation_mat = np.dot(np.array(data).T, np.array(data))
	#relation_mat[7]
	for word in contex_word:#文脈として選んだ言葉のみ抽出
		if(word == "color"):#文脈として色を選んだ場合
			for i in range(100):
				contex_mat.append(relation_mat[i + 7])
				
		contex_mat.append(relation_mat[word_list.index(word)])
	
	contex_vec_list = contex_mat
	#print(contex_mat)
	
	#文脈語群と意味素の内積を計算
	c_hat =[]
	for contex in contex_mat:
		c_tmp = np.matrix(contex) * np.matrix(sem_mat).T
		c_hat.append(c_tmp)
	
	#重心の計算
	#print(len(word_list))
	contex_mat = [0] * 6
	
	for c in c_hat:
		#print(contex_mat)
		contex_mat = np.array(contex_mat) + np.array(c)
		
	contex_mat = contex_mat / np.linalg.norm(contex_mat)
		
	sem_contex = []#文脈を与えた意味行列
	count = 0#カウンター
	#print(contex_mat)
	for i in contex_mat[0]:
		if(i > e):
			#print("Hello")
			sem_contex.append(sem_mat[count])
		count += 1
		
	#もし閾値を超えるような意味素がなければもともとの意味行列を返す	
	if(len(sem_contex) == 0):
		return sem_mat, contex_vec_list
	#print(sem_contex)
	
	else:
		return sem_contex, contex_vec_list
	
#意味空間への射影
def sem_projection(sem_mat, sem_contex, data, contex_vec_list):#dataはデータベースにある画像、input_imgは今回のメイン画像	
	global thresh#閾値以下の距離は0にする処理のための閾値
	#input_vec = np.matrix(input_img) * np.matrix(sem_contex).T
	#data_vec = np.matrix(data) * np.matrix(sem_contex).T
	data_dis =[]#入力データと各データとの距離を記録する
	#print("input:"+str(input_vec))
	#print("data:"+str(data_vec))
	#count = 0
	
	
	#重みｃの計算
	#以下で割る用のベクトルを作成
	div_vec =[]
	for s in sem_mat:
		u = 0
		for c in contex_vec_list:
			u += np.dot(np.array(c) , np.array(s))
		div_vec.append(u)
	
	#すべての文脈語と意味素の内積和をすべての意味素における、すべての文脈語と意味素との内積の和を並べてベクトル化したモノのノルムで割る
	weigth_c = []#各意味素における重みを入れておく箱
	#print(sem_contex)
	#print(contex_vec_list)
	for f in sem_contex:
		w = 0
		for c in contex_vec_list:
			#print(c)
			w += np.dot(np.array(c), np.array(f))
		if(w < 0):#0以下の重みは０に
			w = 0
		weigth_c.append(w / max(div_vec))
	
	print(weigth_c)
	
	#print(len(weigth_c))
	#print(sem_contex)
	#np.array(input_vec)
	#print(input_vec)
	#距離の計算
	#各（文脈から選抜した）意味素において重みを与えて計算する
	#print(data)
	#print(weigth_c)
	cxy = []
	for d in data:
		tmp_cxy = []
		tmps = []
		for d2 in data:
			tmp = np.array(d) - d2
			tmps.append(tmp)
		#print(len(tmps))
		
		for tmp in tmps:
			count = 0
			num = 0
			for t in tmp:
				for w in weigth_c:
					#print(w)
					num += (t * w) ** 2
			
			#print(num)
			num1 =  abs(cmath.sqrt(num))
			tmp_cxy.append(num1)
	
		cxy.append(tmp_cxy)
	
	#print(cxy)
	
	#for d in data.tolist():
	#	count = 0
	#	tmps = []
		
		#print(len(data))
	#	for i in range(len(data)):
	#		tmp = np.array(d) - data[i]
	#		tmps.append(tmp)#なんか二次元配列になっちゃう問題
		
	#	dis = 0
		
	#	for t in tmps:
	#		count = 0
	#		for w in weigth_c:
				#print(tmp)
				#print(w)
	#			tmp[count] *= w
	#			count += 1
		
	#	for i in tmps:
	#		for t in i:
	#			dis += t * t
		
	
			
	#	dis = math.sqrt(dis)
		
	#	if(dis < thresh):
	#		dis = 0
	#		
	#	data_dis.append(dis)
	
	#for d in data_vec:
	#	dis = np.linalg.norm(d - input_vec)
	#	data_dis.append(dis)
		#print(str(count) + " : " + str(dis))
		#count += 1
		
	#print(data_dis)
	return cxy





#
#画像処理
#
#

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
	#re_img = img[ result_rect[1] : result_rect[1] + result_rect[3], result_rect[0] : result_rect[0] + result_rect[2]]

	x = result_rect[0] + int(round(result_rect[2]/2))
	y = result_rect[1] + int(round(result_rect[3]/2))
	#cv.imshow('re_red', re_img)
	g_point = np.array([x, y])
	
	return g_point, result_rect

def detect_figure(img):#重心を使って最短辺から最長辺を求める
	
	global binary_image#2値画像
	
	flag = ""#切創のようなエッジの取りやすい傷でないときに利用
	g_point, rect = find_gravity_r(img)
	tmp_img = edge_detection(img)
	
	min = N
	tmp_point = []
	min_point = []
	min_point1 = []
	min_point2 = []
	max_point1 = []
	max_point2 = []
	dir = 0

	
	#最短辺を求める
	#重心から最も近いエッジを検出
	for i in range(tmp_img.shape[0]):#多分縦方向
		for j in range(tmp_img.shape[1]):#多分横方向
			
			if(tmp_img[i][j].tolist() == 255):#エッジ（白）ならば
				tmp_point = np.array([j, i])#ベクトルを保存

				dir = np.linalg.norm(g_point - tmp_point)
				if(dir < min):
					min = dir
					min_point1 = tmp_point
					
	#エッジが見つからないとき（フラグ持たせて余計な処理させないようにしたほうがいいかも）
	if(len(min_point1) == 0):
		#re_img = img[ result_rect[1] : result_rect[1] + result_rect[3], result_rect[0] : result_rect[0] + result_rect[2]]
		
		#強制的に抽出された傷からすべてを設定します
		
		if(rect[2] >= rect[3]):#x方向に大きいとき
			long_axi = rect[2]
			short_axi = rect[3]
			max_point1 = np.array([rect[0] ,g_point[1]])
			max_point2 = np.array([rect[0] + rect[2], g_point[1]])
			min_point1 = np.array([g_point[0],rect[1]])
			min_point2 = np.array([g_point[0],rect[1] + rect[3]])
			
		else:#y方向に大きいとき
			long_axi = rect[3]
			short_axi = recr[2]
			max_point1 = np.array([g_point[0],rect[1]])
			max_point2 = np.array([g_point[0],rect[1] + rect[3]])
			min_point1 = np.array([rect[0] ,g_point[1]])
			min_point2 = np.array([rect[0] + rect[2], g_point[1]])
			
		flag = "non_openness"
		
		return max_point1, max_point2, min_point1, min_point2, long_axi, short_axi, flag
	
	#重心と一番近いエッジのｘ座標が等しいとき
	elif(g_point[0] == min_point1[0]):
		x = g_point[0]
		
		#cv.drawMarker(img, (min_point1[0], min_point1[1]), (0, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
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
		#print("k is "+str(k)+" , b is"+str(b))
		#print(min_point1)
		#print(g_point)

		#cv.drawMarker(img, (min_point1[0], min_point1[1]), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
		#cv.imshow("min_point",img)		
	
		if(min_point1[0] < g_point[0]):#最短点からみて重心の反対側を探す
			for i in range(tmp_img.shape[0]):
				x = g_point[0] + i
				y = int(k * x + b)
			
				if(x >= tmp_img.shape[1] - 1):
					continue
				elif(y < 0 or y >= tmp_img.shape[0] - 2):
					continue
				
				if(tmp_img[y][x].tolist() == 255 or tmp_img[y][x + 1].tolist() == 255 or tmp_img[y][x - 1].tolist() == 255):#エッジ（白）ならば,幅は３
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
				elif(x < 0 or tmp_img.shape[1] - 1 < x):#幅３なのでその分ｙの範囲が狭まる
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
	
	return max_point1, max_point2, min_point1, min_point2, long_axi, short_axi, flag
	
def detect_edge(img):#形を求める
	tmp_img = edge_detection(img)
	img_edge = img
	max_point1, max_point2, min_point1, min_point2, long_axi, short_axi, flag = detect_figure(img)
	
	g_point = find_gravity_r(img)
	k = (max_point1[1] - max_point2[1])/(max_point1[0] - max_point2[0])
	b = max_point1[1] - k * max_point1[0]
	size = []
	distance = []
	sin_x = []
	sin_y = []
	
	edge_side1 = []
	edge_side2 = []
	#print(max_point1)
	#print(max_point2)
	
	if(abs(k) < 1):#傾きが小さく殆ど水平の時　x= g_point[0]）
		for i in range(int(long_axi)):
			x = max_point1[0] - i
			y = int(round(k * x + b))
			tmp_point1 = []
			tmp_point2 = []
			center = np.array([x, y])
			
			for j in range(int(round(long_axi))):
				x1 = x
				y1 = y + j
				#print(str(x1)+", "+str(y1))
				#cv.drawMarker(img_edge, (x1, y1), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				
				
				if(y1 < 1 or tmp_img.shape[0] - 1 <= y1):
					continue
					
				elif(x1 < 1 or tmp_img.shape[1] - 1 <= x1):
					continue
				
				else:
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1 + 1][x1].tolist() == 255 or tmp_img[y1 - 1][x1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						edge_side1.append(round(np.linalg.norm(tmp_point1 - center),4))
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						edge_side1.append(round(np.linalg.norm(tmp_point1 - center),4))
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
						edge_side2.append(round(np.linalg.norm(tmp_point2 - center),4))
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point2 = np.array([x1, y1])#ベクトルを保存
						edge_side2.append(round(np.linalg.norm(tmp_point2 - center),4))
						break
					
					
			if(len(tmp_point1) != 0 and len(tmp_point2) != 0):
				#print(round(np.linalg.norm(tmp_point1 - tmp_point2)))
				#cv.drawMarker(img_edge, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, (tmp_point1[0], tmp_point1[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, (tmp_point2[0], tmp_point2[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, tuple(max_point1), (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, tuple(max_point2), (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				
				size.append(round(np.linalg.norm(tmp_point1 - tmp_point2), 3))
				distance.append(i)
				
			elif(tmp_point1 == [] or tmp_point2 == []):
				#print("Skip")
				continue
		
		
	
	elif(1 < abs(k) and max_point1[0] > max_point2[0]):
		for i in range(int(long_axi)):
			x = max_point1[0] - i
			y = int(round(k * x + b))
			c = y + 1 / k / x
			tmp_point1 = []
			tmp_point2 = []
			center = np.array([x, y])
			
			for j in range(int(round(long_axi))):
				x1 = int(x + j)
				y1 = int(round(-1 / k * x1 + c))
				#print(str(x1)+", "+str(y1))
				#cv.drawMarker(img_edge, (x1, y1), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				
				
				if(y1 < 1 or tmp_img.shape[0] - 1 <= y1):
					continue
					
				elif(x1 < 1 or tmp_img.shape[1] - 1 <= x1):
					continue
				
				else:
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1 + 1][x1].tolist() == 255 or tmp_img[y1 - 1][x1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						edge_side1.append(round(np.linalg.norm(tmp_point1 - center),4))
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						edge_side1.append(round(np.linalg.norm(tmp_point1 - center),4))
						break
					
				
			for j in range(int(round(long_axi))):
				x1 = int(x - j)
				y1 = int(round(-1 / k * x1 + c))
				#cv.drawMarker(img_edge, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				if(y1 < 1 or tmp_img.shape[0] - 1 <= y1):
					continue
					
				elif(x1 < 1 or tmp_img.shape[1] - 1 <= x1):
					continue
				
				else:
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1 + 1][x1].tolist() == 255 or tmp_img[y1 - 1][x1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point2 = np.array([x1, y1])#ベクトルを保存
						edge_side2.append(round(np.linalg.norm(tmp_point2 - center),4))
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point2 = np.array([x1, y1])#ベクトルを保存
						edge_side2.append(round(np.linalg.norm(tmp_point2 - center),4))
						break
					
					
			if(tmp_point1 != [] and tmp_point2 != []):
				#print(round(np.linalg.norm(tmp_point1 - tmp_point2)))
				#cv.drawMarker(img_edge, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, (tmp_point1[0], tmp_point1[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, (tmp_point2[0], tmp_point2[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, tuple(max_point1), (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, tuple(max_point2), (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				
				size.append(round(np.linalg.norm(tmp_point1 - tmp_point2), 3))
				distance.append(i)
				
			elif(tmp_point1 == [] or tmp_point2 == []):
				#print("Skip")
				continue
			
			
	elif(1 < abs(k) and max_point1[0] <= max_point2[0]):
		for i in range(int(round(long_axi))):		
			x = max_point1[0] + i
			y = int(round(k * x + b))
			c = y + 1 / k / x
			tmp_point1 = []
			tmp_point2 = []
			center = np.array([x, y])
			#cv.drawMarker(img, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
			
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
						edge_side1.append(round(np.linalg.norm(tmp_point1 - center),4))
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point1 = np.array([x1, y1])#ベクトルを保存
						edge_side1.append(round(np.linalg.norm(tmp_point1 - center),4))
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
						edge_side2.append(round(np.linalg.norm(tmp_point2 - center),4))
						break
					if(tmp_img[y1][x1].tolist() == 255 or tmp_img[y1][x1 + 1].tolist() == 255 or tmp_img[y1][x1 -1].tolist() == 255):#エッジ（白）ならば,幅は３
						tmp_point2 = np.array([x1, y1])#ベクトルを保存
						edge_side2.append(round(np.linalg.norm(tmp_point2 - center),4))
						break
					
				
				
			#print(str(tmp_point1) + " , "+ str(tmp_point2))
			
			if(tmp_point1 != [] and tmp_point2 != []):
				#print(round(np.linalg.norm(tmp_point1 - tmp_point2)))
				#cv.drawMarker(img_edge, (x, y), (255, 255, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, max_point1, (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, max_point2, (0, 0, 255), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, (tmp_point1[0], tmp_point1[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				#cv.drawMarker(img_edge, (tmp_point2[0], tmp_point2[1]), (i*15, 255, 25), markerType=cv.MARKER_TILTED_CROSS, markerSize=5)
				size.append(round(np.linalg.norm(tmp_point1 - tmp_point2), 3))
				distance.append(i)
				
			elif(tmp_point1 == [] or tmp_point2 == []):
				print("Skip")
				continue
			
	
	#print(size)
	#cv.drawMarker(img_edge, (g_point[0], g_point[1]), (0, 255, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
	#cv.imshow("edge",img_edge)
	#cv.imshow("img",tmp_img)
	return size, distance, short_axi, edge_side1, edge_side2, flag
	

def oval_judge(img):
	size, distance, short_axi, e1, e2, flag = detect_edge(img)
	
	if(flag == "non_openness"):#非開放性の時
		return 0
	
	#傷が円に近いほどsinとの誤差が小さくなる
	sin_x = np.arange(0,distance.index(max(distance))+1, 1)
	sin_y = np.sin(sin_x/(distance.index(max(distance)))*math.pi)*short_axi
	
	#差分は正規化
	difference = abs((size - sin_y)*(size - sin_y))/short_axi/short_axi
	plt.scatter(distance, size, label="difference", color="red")
	plt.plot(sin_x, sin_y, label="difference", color="green")
	#plt.show()
	difference = sum(difference)/len(difference)#平均
	print("Oval:"+str(round(difference,2)))
	
	if (difference <= 0.02):
		return 1
	else:
		return 0

def edge_judge(img):#創縁不整と創縁直線を判定
		size, distance, short_axi, edge_side1, edge_side2, flag = detect_edge(img)
		
		if(flag == "non_openness"):#非開放性の時
			return 1, -1
		
		edge_irregular = 0
		edge_straight = 0
		
		#三分割にする
		sp_size1 = list(np.array_split(edge_side1, 3))
		md_size1 = sp_size1[1]
		
		sp_size2 = list(np.array_split(edge_side2, 3))
		md_size2 = sp_size2[1]
		
		
		#標準偏差、平均、偏導関数を計算
		ev1 = stdev(md_size1)
		ave1 = mean(md_size1)
		cv1 = ev1 / ave1#相対的なバラつきを計算
		
		ev2 = stdev(md_size2)
		ave2 = mean(md_size2)
		cv2 = ev2 / ave2#相対的なバラつきを計算
		
		#創縁不整,直線を定義
		if(cv1 < 0.1 and cv2 < 0.1):
			edge_straight = 1
			edge_irregular = -1
		elif(cv1 >= 0.1 and cv2 >= -0.1):
			edge_straight = -1
			edge_irregular = 1
		else:
			edge_straight = 0
			edge_irregular = 1
		#print("cv : "+ str(cv))
		
		
		return edge_irregular, edge_straight

def sharp_judge(img):
	size, distance, short_axi, e1, e2, flag = detect_edge(img)
	
	if(flag == "non_openness"):#非開放性の時
		return 0 ,0, -1
	
	
	
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
	#plt.show()
	
	if(fw_result < 0.5 and bw_result < 0.5):#どちらの端も0.5未満なら創端鋭利
		return 1, 0, 1
		
	elif(fw_result >= 0.5 and bw_result >= 0.5):#どちらの端も0.5未満なら創端太
		return 0, 1, 1
		
	elif((fw_result < 0.5 or bw_result < 0.5) and (fw_result >= 0.5 and bw_result >= 0.5)):#どちらかの端点が太く、もう一方が鋭利
		return 1, 1, 1
		
	else:
		return 0, 0, 1
		

def contrast(image, a):#(aはゲイン)
	image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)#グレースケールに変換
	lut = [ np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.))) for i in range(256)] 
	result_image = np.array( [ lut[value] for value in image.flat], dtype=np.uint8 )
	result_image = result_image.reshape(image.shape)
	return result_image


#重心を見つける関数の使いまわし、傷周辺の長方形だけを繰りぬくように改変
def find_wound_COLOR(img):#HSVカラーモデルから重心を探す
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

	if(rects == []):
		return img
	#最大の四角を見つける
	result_rect = max(rects, key=(lambda x: x[2] * x[3]))

	#cv.rectangle(img, tuple(result_rect[0:2]), tuple(result_rect[0:2] + result_rect[2:4]), (0, 255, 0), thickness=2)
	#print(result_rect)
	re_img = img[ result_rect[1] : result_rect[1] + result_rect[3], result_rect[0] : result_rect[0] + result_rect[2]]
	x = result_rect[0] + int(round(result_rect[2]/2))
	y = result_rect[1] + int(round(result_rect[3]/2))
	#cv.imshow('re_red', re_img)
	g_point = np.array([x, y])

	return re_img

#画像を読み込んでLab空間に変換
def toLab(img):
	cols = 120 #ヒストグラムの行と列の数
	
	img_ori = find_wound_COLOR(img)#傷周辺のみを切り抜いた画像
	img_Lab = cv.cvtColor(img_ori, cv.COLOR_BGR2Lab)
	#print(img_Lab)
	img_L, img_a, img_b = cv.split(img_Lab)
	
	#明度の解析
	img_L = np.ndarray.flatten(img_L)
	#print(int(mean(img_L.tolist()) / 256 * 100))
	brightness = int(mean(img_L.tolist()) / 256 * 100)
	
	#プロットしてみる
	img_a = np.ndarray.flatten(img_a)
	img_b = np.ndarray.flatten(img_b)
	#print(img_a)
	hist, aedges, bedges= np.histogram2d(img_a, img_b, bins=cols, range=[[0,255],[0,255]])
	
	#print(np.array(hist.tolist()).shape)
	
	
	#一度一次元配列に変換してから二次元での位置を確認する
	tmp_hist = np.ndarray.flatten(hist)
	
	#tmp_histで大きな値を10こもってくる
	max_list1 = []
	for i in range(1,30):
		num = sorted(tmp_hist)[-i]
		if(num == 0):
			continue
			
		indexes = [j for j, z in enumerate(tmp_hist) if z == num]
		
		for place in indexes:
			x = int(place / cols)
			y = place % cols
		
			max_list1.append(np.array([x, y]))
		
	
	#print(max_list1)
	
	
	#hist_list = hist.tolist()
	#max_list = max(hist_list)
	#m = max(max_list)
	
	#ヒストグラムで一番多きいところの座標を習得 : hist[y][x]
	#x = hist_list.index(max(hist_list))
	#y = max_list.index(m)
	
	degs = []
	saturations = []
	opt = ""
	for val in max_list1:
		x = val[0]
		y = val[1]
	
		#原点が（cols/2,cols / 2）担っているのでこれを（0,0）にシフトし、赤を0度として角度で色情報を付与
		x -= int(cols / 2)
		y -= int(cols / 2)
		r = math.sqrt(x**2 + y**2)
		cos = x / r
		sin = y / r
		r = int(r)
		#print(r)
		#print(math.degrees(math.acos(cos)))
		#print(math.degrees(math.asin(sin)))
		deg = math.degrees(math.acos(cos))
	
		if(math.degrees(math.asin(sin)) < 0):#yがマイナスなら下半分
			deg = -1 * math.fabs(deg)
		
		if(brightness >= 60):
			opt = 'pale'
			
		elif(brightness <= 40):
			opt = 'dark'
		
		elif(r <= 20):
			opt = 'graylsh'
		
		elif(20 < r and r <= 40):
			opt = 'dull'
			
		else:
			opt = ''
			
		degs.append([deg, opt])
	
	#色の判定
	color_list = []	
	for deg in degs:	
		if(0 <= deg[0] and deg[0] < 10):
			color_list.append(deg[1] + "_10RP")
	
		elif(10 <= deg[0] and deg[0] < 29):
			color_list.append(deg[1] +"_5R")
	
		elif(29 <= deg[0] and deg[0] < 47):
			color_list.append(deg[1] +"_10R")
		
		elif(47 <= deg[0] and deg[0] < 65):
			color_list.append(deg[1] +"_5YR")
		
		elif(65 <= deg[0] and deg[0] < 83):
			color_list.append(deg[1] +"_10YR")
		
		elif(83 <= deg[0] and deg[0] < 101):
			color_list.append(deg[1] +"_5Y")
		
		elif(101 <= deg[0] and deg[0] < 119):
			color_list.append(deg[1] +"_10Y")
		
		elif(119 <= deg[0] and deg[0] < 137):
			color_list.append(deg[1] +"_5GY")
		
		elif(137 <= deg[0] and deg[0] < 155):
			color_list.append(deg[1] +"_10GY")
		
		elif(155 <= deg[0] and deg[0] < 173):
			color_list.append(deg[1] +"_5G")
	
		elif(0 > deg[0] and deg[0] > -10):
			color_list.append(deg[1] +"_10RP")
	
		elif(-10 >= deg[0] and deg[0] > -29):
			color_list.append(deg[1] +"_5RP")
	
		elif(-29 >= deg[0] and deg[0] > -47):
			color_list.append(deg[1] +"_10P")
		
		elif(-47 >= deg[0] and deg[0] > -65):
			color_list.append(deg[1] +"_5P")
		
		elif(-65 >= deg[0] and deg[0] > -83):
			color_list.append(deg[1] +"_10PB")
		
		elif(-83 >= deg[0] and deg[0] > -101):
			color_list.append(deg[1] +"_5PB")
		
		elif(-101 >= deg[0] and deg[0] > -119):
			color_list.append(deg[1] +"_10B")
		
		elif(-119 >= deg[0] and deg[0] > -137):
			color_list.append(deg[1] +"_5B")
		
		elif(-137 >= deg[0] and deg[0] > -155):
			color_list.append(deg[1] +"_10BG")
		
		elif(-155 >= deg[0] and deg[0] > -173):
			color_list.append(deg[1] +"_5BG")
	
		else:
			color_list.append(deg[1] +"_10G")
		
	
	#print(list(set(color_list)))
	
	#img = cv.imread(hist)
	#cv.imshow()

	#histに64*64のマスに値が入ってます
	plt.figure()
	sns.heatmap(hist, cmap="binary_r")
	plt.title("Histgram 2D")
	plt.xlabel("a*")
	plt.ylabel("b*")
	plt.savefig('D:\Sotsuken\Sotsuken_repo\output\heat_map.png')
	
	

	#x,y座標を３Dの形式に変換
	#apos, bpos = np.meshgrid(aedges[:-1], bedges[:-1])
	#zpos = 0#zは０を始点にする

	#x,y座標の幅を指定
	#da = apos[0][1] - apos[0][0]
	#db = bpos[1][0] - bpos[0][0]
	#dz = hist.ravel()

	#x,yを３Dの形に変換
	#apos = apos.ravel()
	#bpos = bpos.ravel()

	#３D描画
	#fig = plt.figure()#描画領域の作成
	#aa = fig.add_subplot(111, projection="3d")
	#aa.bar3d(apos, bpos, zpos, da, db, dz, cmap=cm.hsv)#ヒストグラムを３D空間に表示
	#plt.title("Histgram 2D")
	#plt.xlabel("a*")
	#plt.ylabel("b*")
	#aa.set_zlabel("Z")
	#plt.show()

	src1 = cv.imread('D:\Sotsuken\Sotsuken_repo\output\heat_map.png')
	src2 = cv.imread('D:\Sotsuken\Sotsuken_repo\output\Lab2.jpg')
	#cv.imshow('src', src1)
	cv.rectangle(src1, (70, 50), (490, 440), (255, 0, 255), thickness=8, lineType=cv.LINE_4)

	rect = (80,58, 397, 368)
	src1 = src1[ rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
	#色の表示系がずれているので回転
	src1 = cv.rotate(src1, cv.ROTATE_90_COUNTERCLOCKWISE)

	src1 = cv.resize(src1, src2.shape[1::-1])
	dst = cv.addWeighted(src1, 0.5, src2, 0.5, 0)

	#cv.imshow('result.jpg', dst)
	#cv.waitKey()
	return list(set(color_list)), dst
	
def color_judge(color_list):
	palette = [0]*100

	for color in color_list:
		opt = color.split("_")
	
		num1 = 0
		num2 = 0
		#opt[0]がオプションopt[1]が色相
		if(opt[0] == 'pale'):
			num1 = 0
	
		elif(opt[0] == 'graylsh'):
			num1 = 1
		
		elif(opt[0] == 'dull'):
			num1 = 2
		
		elif(opt[0] == 'dark'):
			num1 = 3
	
		else:
			num1 = 4
	
		#ここから色相
		if(opt[1] == '5Y'):
			num2 = 0
	
		elif(opt[1] == '10YR'):
			num2 = 1
		
		elif(opt[1] == '5YR'):
			num2 = 2
		
		elif(opt[1] == '10R'):
			num2 = 3
	
		elif(opt[1] == '5R'):
			num2 = 4
	
		elif(opt[1] == '10RP'):
			num2 = 5
	
		elif(opt[1] == '5RP'):
			num2 = 6
		
		elif(opt[1] == '10P'):
			num2 = 7

		elif(opt[1] == '5P'):
			num2 = 8
		
		elif(opt[1] == '10PB'):
			num2 = 9
		
		elif(opt[1] == '5PB'):
			num2 = 10
		
		elif(opt[1] == '10B'):
			num2 = 11
		
		elif(opt[1] == '5B'):
			num2 = 12
		
		elif(opt[1] == '10BG'):
			num2 = 13
		
		elif(opt[1] == '5BG'):
			num2 = 14
		
		elif(opt[1] == '10G'):
			num2 = 15
		
		elif(opt[1] == '5G'):
			num2 = 16
	
		elif(opt[1] == '10GY'):
			num2 = 17
		
		elif(opt[1] == '5GY'):
			num2 = 18
		
		else:#10Y
			num2 = 19
		
		num3 = num2 * 5 + num1
		palette[num3] = 1
		
	return palette



#pullpush判定は使わないかも
def pullpush_judge(img):#文字列でpullかpushかを返します
	global light_point, kaizoudo
	gravity = find_gravity_r(img)#重心
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
	#cv.imshow("ligth", img_list[num])
	
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
	#cv.drawMarker(tmp_img, (x, y), (0, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
	
	if(gravity[0] != light_point[0]):
		k = (gravity[1] - light_point[1]) /(gravity[0] - light_point[0]) 
		b = gravity[1] - k * gravity[0]
	
	edge_img = edge_detection(img)
	edge_point = []
	
	#print(gravity)
	#print(light_point)
	#cv.drawMarker(tmp_img, tuple(light_point), (30, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)

	if(gravity[0] == light_point[0] and gravity[1] > light_point[1]):#縦に並んだ時
		for i in range(height):
			x = gravity[0]
			y = gravity[1] + i
		
			if(x < 0 or y < 0 or edge_img.shape[1] < x or edge_img.shape[0] < y):
				continue
		
			#cv.drawMarker(tmp_img, (x, y), (30, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)

			if(edge_img[y][x].tolist() == 255 or edge_img[y+1][x].tolist() == 255 or edge_img[y-1][x].tolist() == 255):
				#print(str(x)+" , "+str(y))
				#cv.drawMarker(tmp_img, (x, y), (0, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
				edge_point = np.array([x, y])
				break
			
		if(len(edge_point) == 0):
			return "Error"
			
		for i in range(int(10*img.shape[0]*img.shape[1]/40000)):
			x1 = edge_point[0]
			y1 = edge_point[1] + i
			x2 = edge_point[0] - 1
			y2 = edge_point[1] + i
			x3 = edge_point[0] + 1
			y3 = edge_point[1] + i
		
			result = round((tmp_img[y1][x1] + tmp_img[y2][x2] + tmp_img[y3][x3])/3, 3)
			result_list.append(result)
	
	elif(gravity[0] == light_point[0] and gravity[1] <= light_point[1]):#縦に並んだ時
		for i in range(height):
			x = gravity[0]
			y = gravity[1] - i
		
			if(x < 0 or y < 0 or edge_img.shape[1] < x or edge_img.shape[0] < y):
				continue
		
			#cv.drawMarker(tmp_img, (x, y), (30, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)

			if(edge_img[y][x].tolist() == 255 or edge_img[y+1][x].tolist() == 255 or edge_img[y-1][x].tolist() == 255):
				#print(str(x)+" , "+str(y))
				#cv.drawMarker(tmp_img, (x, y), (0, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
				edge_point = np.array([x, y])
				break
			
		if(len(edge_point) == 0):
			return "Error"
			
		for i in range(int(10*img.shape[0]*img.shape[1]/40000)):
			x1 = edge_point[0] + i
			y1 = edge_point[1]
			x2 = edge_point[0] + i
			y2 = edge_point[1] - 1
			x3 = edge_point[0] + i
			y3 = edge_point[1] + 1
		
			result = round((tmp_img[y1][x1] + tmp_img[y2][x2] + tmp_img[y3][x3])/3, 3)
			result_list.append(result)
		
	elif(abs(k) < 1 and gravity[0] < light_point[0]):#横に並んだ時
		for i in range(height):
			x = gravity[0] - i
			y = gravity[1]
		
			if(x < 0 or y < 0 or edge_img.shape[1] < x or edge_img.shape[0] < y):
				continue
		
			#cv.drawMarker(tmp_img, (x, y), (30, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)

			if(edge_img[y][x].tolist() == 255 or edge_img[y+1][x].tolist() == 255 or edge_img[y-1][x].tolist() == 255):
				#print(str(x)+" , "+str(y))
				#cv.drawMarker(tmp_img, (x, y), (0, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
				edge_point = np.array([x, y])
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
			
	elif(abs(k) < 1 and gravity[0] >= light_point[0]):#横に並んだ時
		for i in range(height):
			x = gravity[0] + i
			y = gravity[1]
		
			if(x < 0 or y < 0 or edge_img.shape[1] < x or edge_img.shape[0] < y):
				continue
		
			#cv.drawMarker(tmp_img, (x, y), (30, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)

			if(edge_img[y][x].tolist() == 255 or edge_img[y+1][x].tolist() == 255 or edge_img[y-1][x].tolist() == 255):
				#print(str(x)+" , "+str(y))
				#cv.drawMarker(tmp_img, (x, y), (0, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
				edge_point = np.array([x, y])
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
	
	else:
		for i in range(height):
			if(gravity[0] >= light_point[0]):
				x = gravity[0] + i
			else:
				x = gravity[0] - i
			y = int(round(k * x + b))
			if(x < 0 or y < 1 or edge_img.shape[1] < x or edge_img.shape[0] < y):
				continue
		
			#cv.drawMarker(tmp_img, (x, y), (30, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
			
			if(edge_img[y][x].tolist() == 255 or edge_img[y+1][x].tolist() == 255 or edge_img[y-1][x].tolist() == 255):
				#print(str(x)+" , "+str(y))
				#cv.drawMarker(tmp_img, (x, y), (0, 0, 0), markerType=cv.MARKER_TILTED_CROSS, markerSize=15)
				edge_point = np.array([x, y])
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
	#cv.imshow("tmp_img",tmp_img)
	#cv.imshow("img",img)
	#plt.show()
	
	if(num < 0):#上に凸なら
		return "push"
		
	else:
		return "pull"


def judge(id, img):
	end_sharp = 0
	end_thick = 0
	edge_irregular = 0
	edge_straight = 0
	oval = 0
	openness = 0
	non_openness = 0
	
	detect_figure(img)
	detect_edge(img)
	
	#エッジのイメージを代入
	edge_img = edge_detection(img)
	
	#創傷端を判定
	end_sharp, end_thick, openness = sharp_judge(img)
	
	#創傷縁を判定
	edge_irregular, edge_straight = edge_judge(img)
	
	#円度を判定
	oval = oval_judge(img)
	
	#色の判定
	color_list, color_hist_img = toLab(img)
	palette = color_judge(color_list)
	
	#print("創端鋭利："+str(end_sharp)+"　創端太："+str(end_thick)+" 創縁不整："+str(edge_irregular)+" 創縁直線："+str(edge_straight)+"　円度："+ str(oval)
	#+"　開放性："+ str(openness)+"　非開放性："+ str(non_openness))
	
	
	#画像データをまとめる(画像は書き出してパスをわたすことにした)
	result_path = 'D:\\Sotsuken\\Sotsuken_repo\\result\\'
	path1 = str(result_path)+ 'original_img\\ori_img_' + str(id) + '.jpg'
	path2 = str(result_path)+ 'edge_img\\edge_img_'+str(id)+'.jpg'
	path3 = str(result_path)+ 'color_hist_img\\color_hist_img_'+str(id)+'.jpg'
	
	cv.imwrite(path1, img)
	cv.imwrite(path2, edge_img)
	cv.imwrite(path3, color_hist_img)
	
	#今後書き直す
	if(edge_straight == 1):
		edge_irregular = 0
	
	
	#01で特徴を示す場合
	result = [end_sharp,end_thick,edge_irregular,edge_straight,oval,openness,non_openness]
	#特徴ベクトルを正規化する場
		
	#print(result)
	
	#色情報と合成
	#result.extend(palette)
	
	#辞書作成
	data = {'original_img' : path1, 'edge_img':path2,
				'color_hist_img':path3, 'color': color_list}


	
	return result,data
	
	
def read_img(folder):#フォルダを指定して
	files = glob.glob(folder)
	results = []
	data_list = []
	id = 0
	
	for file in files:
		#print(file)
		img = cv.imread(file, cv.IMREAD_COLOR)
		#sharp, oval, pull, push = judge(img)
		result, data = judge(id, img)
		data_list.append(data)
		results.append(result)
		id += 1
	
		#print(data_list)
	with open('D:\\Sotsuken\\Sotsuken_repo\\result\\output_file\\img_infos.csv', 'w') as f:
		writer = csv.DictWriter(f, ['original_img', 'edge_img', 'color_hist_img', 'color'])
		#ヘッダの書き込み
		writer.writeheader()
		
		for data in data_list:
			#print(data)
			writer.writerow(data)
			
	with open('D:\\Sotsuken\\Sotsuken_repo\\result\\output_file\\img_vec.csv', 'w') as f:
		writer2 = csv.writer(f)
		for row in results:
			writer2.writerow(row)
		
	return results
	
	
def mmm_operation(path):
	results = read_img(path)
	#data_listhは画像のパスまで入っている
	#resultsは単純にベクトルのみ
	#print(data_list)
	
	#word_listに色追加しなくちゃいけない、、、。
	word_list = ["end_sharp","end_thick","edge_irregular","edge_straight","oval","openness"]
	#data_mat = np.array(results)
	sem_mat = make_semantic_matrix(results)
	#sem_data = cl.OrderedDict()
	
	#意味空間をｃｓｖで出力
	with open('D:\\Sotsuken\\Sotsuken_repo\\result\\output_file\\sem_mat.csv', 'w') as f:
		writer = csv.writer(f)
		#writer.writerow(word_list)
		for row in sem_mat:
			writer.writerow(row)
	
	#まずすべての文脈において距離計算
	word_list = ["end_sharp","end_thick","edge_irregular","edge_straight","oval","openness"]
	contex_word = [["end_sharp","end_thick","edge_irregular","edge_straight","oval","openness"]]
	
	#文脈の種類を作成
	contex_list = []
	c_list = ["incision", "contusion", "all"]#文脈の順番を格納
	incision_contex = ["end_sharp", "edge_straight", "openness"]
	contusion_contex = ["edge_irregular","openness"]
	all_context = word_list
	contex_list.append(incision_contex)
	contex_list.append(contusion_contex)
	contex_list.append(all_context)
	
	sem_contex, contex_vec_list = make_context(sem_mat, word_list, contex_word, data_mat)
	
	count = 0
	for contex_word in contex_list:#全てのコンテクストについて距離を計算しdistance_listに格納
		distances_list = []
		distances = []#各画像から画像までの距離
		
		sem_contex, contex_vec_list = make_context(sem_mat, word_list, contex_word, data_mat)
		
		#for img_vec in results:#画像毎にdataとの距離計算
		#print(img_vec)
		distances = sem_projection(sem_mat, sem_contex, results, contex_vec_list)
		#distances_list.append(distances)
		
		#print(distances)
		
		with open('D:\\Sotsuken\\Sotsuken_repo\\result\\output_file\\context_dist\\'+str(count + 1)+'_'+c_list[count]+'_context.csv', 'w') as f:
			writer = csv.writer(f)
			for d in distances:
				writer.writerow(d)
		count += 1

		
	with open('D:\\Sotsuken\\Sotsuken_repo\\result\\output_file\\context_dist\\0_no_context.csv', 'w') as f:
		writer = csv.writer(f)
		#writer.writerow(no_context_dist(results))	
		for d in no_context_dist(results):
				writer.writerow(d)
		
		
	#文脈毎にcsvで出力
		
	
	#print(data_list)


if __name__ == '__main__':

	#意味空間を構成するための画像群（のちのLMMLファイル）が入っているフォルダのパスを渡す
	mmm_operation(path)
	
	#実際に距離計算を行いたい画像群の（のちのLMMLファイル）を渡す
	#similarity_operation(path)
	
	
	
	
		#img = cv.imread(path)
		#img_edge = cv.imread(path)
		#judge(img)
		
		
		
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
