import numpy as np
import pandas as pd

def make_semantic_matrix(data_mat):#意味行列を作るためのに作成したセマンティックな行列を入力
	
	#相関行列の作成
	relation_mat = np.dot(np.array(data_mat).T, np.array(data_mat))
	eig_val, eig_vec = np.linalg.eig(relation_mat)#固有値と固有ベクトルを取得
	#print(eig_val)
	#print(eig_vec)
	
	return eig_vec
	

#文脈の作成
def make_context(sem_mat, word_list, contex_word):#意味行列に文脈を指定する単語リストを入力
	contex_mat = [0] * len(word_list)
	for word in contex_word:#文脈として選んだ言葉のみ抽出
		contex_mat[word_list.index(word)] = 1
	
	#print(contex_mat)
	
	sem_contex = []#文脈を与えた意味行列
	count = 0#カウンター
	for i in contex_mat:
		if(i == 1):
			sem_contex.append(sem_mat[count])
		count += 1
	#print(sem_contex)
	return sem_contex
	
#意味空間への射影
def sem_projection(sem_contex, data, input_img):#dataはデータベースにある画像、input_imgは今回のメイン画像
	input_vec = np.matrix(input_img) * np.matrix(sem_contex).T
	data_vec = np.matrix(data) * np.matrix(sem_contex).T
	data_dis =[]#入力データと各データとの距離を記録する
	#print("input:"+str(input_vec))
	#print("data:"+str(data_vec))
	#count = 0
	for d in data_vec:
		dis = np.linalg.norm(d - input_vec)
		data_dis.append(dis)
		#print(str(count) + " : " + str(dis))
		#count += 1
		
	return data_dis
	
	
if __name__ == '__main__':
	data = [[1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 0]]#模範データ(今の段階ではデータベースの既存データとしても利用中)
	word_list = ["sharp", "oval", "pull", "push"]
	contex_word = ["sharp"]
	
	sem_mat = make_semantic_matrix(data)
	sem_contex = make_context(sem_mat, word_list, contex_word)
	
	input_img = [0 ,0 ,0 ,1]#テスト用の入力画像
	data_dis = sem_projection(sem_contex, data, input_img)
	print(data_dis)
	