#codeing:utf-8
# -*- coding: utf-8 -*-

import os, sys, time
import cv2 as cv
from PIL import Image
from PIL import ImageDraw
import numpy as np
np.set_printoptions(threshold=np.inf)

import math

import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


import glob
from statistics import mean, stdev, mode

plt.style.use('ggplot')
plt.rcParams["axes.facecolor"] = "white"

import plotly.graph_objects as go


path = "D:\Sotsuken\Sotsuken_repo\sample\\bruise"

#研究室のときはこちらを利用
#path = ".\sample\\incision_1.jpg"


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
			color_list.append(deg[1] + "10RP")
	
		elif(10 <= deg[0] and deg[0] < 29):
			color_list.append(deg[1] +"5R")
	
		elif(29 <= deg[0] and deg[0] < 47):
			color_list.append(deg[1] +"10R")
		
		elif(47 <= deg[0] and deg[0] < 65):
			color_list.append(deg[1] +"5YR")
		
		elif(65 <= deg[0] and deg[0] < 83):
			color_list.append(deg[1] +"10YR")
		
		elif(83 <= deg[0] and deg[0] < 101):
			color_list.append(deg[1] +"5Y")
		
		elif(101 <= deg[0] and deg[0] < 119):
			color_list.append(deg[1] +"10Y")
		
		elif(119 <= deg[0] and deg[0] < 137):
			color_list.append(deg[1] +"5GY")
		
		elif(137 <= deg[0] and deg[0] < 155):
			color_list.append(deg[1] +"10GY")
		
		elif(155 <= deg[0] and deg[0] < 173):
			color_list.append(deg[1] +"5G")
	
		elif(0 > deg[0] and deg[0] > -10):
			color_list.append(deg[1] +"10RP")
	
		elif(-10 >= deg[0] and deg[0] > -29):
			color_list.append(deg[1] +"5RP")
	
		elif(-29 >= deg[0] and deg[0] > -47):
			color_list.append(deg[1] +"10P")
		
		elif(-47 >= deg[0] and deg[0] > -65):
			color_list.append(deg[1] +"5P")
		
		elif(-65 >= deg[0] and deg[0] > -83):
			color_list.append(deg[1] +"10PB")
		
		elif(-83 >= deg[0] and deg[0] > -101):
			color_list.append(deg[1] +"5PB")
		
		elif(-101 >= deg[0] and deg[0] > -119):
			color_list.append(deg[1] +"10B")
		
		elif(-119 >= deg[0] and deg[0] > -137):
			color_list.append(deg[1] +"5B")
		
		elif(-137 >= deg[0] and deg[0] > -155):
			color_list.append(deg[1] +"10BG")
		
		elif(-155 >= deg[0] and deg[0] > -173):
			color_list.append(deg[1] +"5BG")
	
		else:
			color_list.append(deg[1] +"10G")
		
	
	print(list(set(color_list)))
	
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
	#cv.rectangle(src1, (70, 50), (490, 440), (255, 0, 255), thickness=8, lineType=cv.LINE_4)

	rect = (80,58, 397, 368)
	src1 = src1[ rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
	#色の表示系がずれているので回転
	src1 = cv.rotate(src1, cv.ROTATE_90_COUNTERCLOCKWISE)

	src1 = cv.resize(src1, src2.shape[1::-1])
	dst = cv.addWeighted(src1, 0.5, src2, 0.5, 0)

	#cv.imshow('result.jpg', dst)
	#cv.waitKey()
	return dst
	
def read_img(folder):#フォルダを指定して
	files = glob.glob(os.path.join(path, '*.jpg'))
	result_list = []
	name_list = ["bruise_1", "bruise_2", "bruise_3", "health_1","health_2", "test", "test2","test3"]
	count = 0
	for file in files:
		print(file)
		img = cv.imread(file)
		result = toLab(img)
		result_list.append(result)
		count += 1
	
	#画像を表示
	#for i in range(len(result_list)):
#		cv.imshow(name_list[i], result_list[i])
#	cv.waitKey()


if __name__ == '__main__':
	read_img(path)
