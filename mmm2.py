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
import copy

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
	


def make_context(sem_mat, word_list, contex_word, data):
	contex_mat = []
	context_vec_list = []
	count = 0
	
	for c in contex_word:
		contex_mat = [0] * len(sem_mat[0])
		for word in c:#文脈として選んだ言葉のみ抽出
			#if(word == "color"):#文脈として色を選んだ場合
				#for i in range(100):
					#contex_mat.append(relation_mat[i + 7])
			index = word_list.index(word)
			print("index : "+str(index))
			contex_mat[index] = 1
			context_vec_list.append(contex_mat)
	
	return context_vec_list

#変な関数
def make_context1(sem_mat, word_list, contex_word, data):#意味行列に文脈を指定する単語リストを入力
	global e
	contex_mat = []
	count = 0
	
	contex_mat = [0] * (len(sem_mat[0]))
	for word in contex_word:#文脈として選んだ言葉のみ抽出
		if(word == "color"):#文脈として色を選んだ場合
			for i in range(100):
				contex_mat.append(relation_mat[i + 7])
				
		
	
	contex_vec_list = contex_mat
	#print(contex_mat)
	
	#文脈語群と意味素の内積を計算
	c_hat =[]
	for contex in contex_mat:
		c_tmp = np.matrix(contex) * np.matrix(sem_mat).T
		c_hat.append(c_tmp)
	
	#重心の計算
	#print(len(word_list))
	contex_mat = [0] * (len(sem_mat[0]))
	
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
def sem_projection(sem_mat, data, contex_vec_list):#dataはデータベースにある画像、input_imgは今回のメイン画像	
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
	max = 0
	for s in sem_mat:
		u = 0
		for c in contex_vec_list:
			u += np.dot(np.array(c) , np.array(s))
		div_vec.append(u)
	
	
	for num in div_vec:
		#print(max)
		#print(abs(num))
		if(abs(max) < abs(num)):
			max = num
		
	
	print("div_vec : ")
	print(max)
	
	#意味重心ベクトル
	G = []
	tmp_sem = []#sem_matを書き換える
	for s in sem_mat:
		w = 0
		for c in contex_vec_list:
			w += np.dot(np.array(c), np.array(s))
		tmp = s
		if(w < 0):
			tmp = -1 * s
			print(tmp)
		tmp_sem.append(tmp)
		G.append(w / max)
	print("Gravity")
	print(G)
	
	#すべての文脈語と意味素の内積和をすべての意味素における、すべての文脈語と意味素との内積の和を並べてベクトル化したモノを最大ノルムで割る
	weigth_c = []#各意味素における重みを入れておく箱
	#print(sem_contex)
	#print(contex_vec_list)
	count = 0
	for s in sem_mat:
		w = 0
		for c in contex_vec_list:
			#print(w)
			w += np.dot(np.array(c), np.array(s))
			
		if(abs(w) < 0.0001):
			w  = 0
		weigth_c.append(w / max)
		count += 1
	print("weight")
	print(weigth_c)
	print("tmp_sem")
	print(tmp_sem)
	
	#print(len(weigth_c))
	#print(sem_contex)
	#np.array(input_vec)
	#print(input_vec)
	#距離の計算
	#各（文脈から選抜した）意味素において重みを与えて計算する
	#print(data)
	#print(weigth_c)
	cxy = []
	data_vec=[]
	for d in data:#データをベクトルに変換
		#print(d)
		d_vec=[]
		count = 0
		for s in tmp_sem:
			#print(weigth_c[count])
			count1 = 0
			tmp = 0
			for i in d:#dataの成分が１の時内積を計算
				if(i == 1):
					tmp1 = [0] * (len(sem_mat[0]))
					tmp1[count1] = 1
					print(tmp1)
					if(np.dot(np.array(tmp1), np.array(s)) < 0):#本当は＜がいい
						count1 += 1
						continue
					tmp += (np.dot(np.array(tmp1), np.array(s)))*weigth_c[count]
					
				count1 += 1
			d_vec.append(tmp)
		data_vec.append(d_vec)
		
	print("data_vec")
	print(data_vec)
		
	#距離計算
	for d1 in data_vec:
		tmp_cxy = []
		for d2 in data_vec:
			tmp = np.linalg.norm(np.array(d1) - np.array(d2))
			tmp_cxy.append(tmp)
		cxy.append(tmp_cxy)
			
	
	
	#for d1 in data:
	#	tmp_cxy = []
	#	for d2 in data:
	#		count = 0
	#		sum = 0
	#		print("内積")
	#		for s in sem_mat:
	#			tmp1 = np.dot(np.array(d1), np.array(s))
	#			tmp2 = np.dot(np.array(d2), np.array(s))
	#			tmp3 = tmp1 - tmp2#x-y
				#print(tmp1)
	#			tmp3 *= weigth_c[count]#c(x-y)
	#			tmp3 = tmp3 ** 2
	#			print(tmp3)
	#			sum += tmp3
	#			print(sum)
	#			count += 1
			
	#		sum = abs(cmath.sqrt(sum))
	#		tmp_cxy.append(sum)
			
	#	cxy.append(tmp_cxy)
	print("distance")		
	print(cxy)
	
	
	
	#for d in data:
	#	tmp_cxy = []
	#	tmps = []
	#	for d2 in data:
	#		tmp = np.array(d) - d2
	#		tmps.append(tmp)
		#print(len(tmps))
		
	#	for tmp in tmps:
	#		count = 0
	#		num = 0
	#		for t in tmp:
	#			for w in weigth_c:
	#				#print(w)
	#				num += (t * w) ** 2
	#		
	#		#print(num)
	#		num1 =  abs(cmath.sqrt(num))
	#		tmp_cxy.append(num1)
	#
	#	cxy.append(tmp_cxy)

	return cxy


if __name__ == '__main__':
	d = [[1,1],[1, 0], [0, 1]]#模範データ(今の段階ではデータベースの既存データとしても利用中)
	data = [[1, 1],[1,0],[0,1]]
	word_list = ["soft", "hard"]
	contex_word = [["soft"]]
	sem_mat = make_semantic_matrix(d)
	print("sem_mat")
	print(sem_mat)
	contex_vec = make_context(sem_mat, word_list, contex_word, data)
	#input_img = [1 ,0 ,0 ,1]#テスト用の入力画像
	print("context")
	print(contex_vec)
	
	data_dis = sem_projection(sem_mat, data, contex_vec)
	