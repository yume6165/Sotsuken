import numpy as np
import pandas as pd
import math

#閾値
e = -1


def flatten(data):
    for item in data:
        if hasattr(item, '__iter__'):
            for element in flatten(item):
                yield element
        else:
            yield item

def make_semantic_matrix(data_mat):#意味行列を作るためのに作成したセマンティックな行列を入力
	
	#相関行列の作成
	relation_mat = np.dot(np.array(data_mat).T, np.array(data_mat))
	eig_val, eig_vec = np.linalg.eig(relation_mat)#固有値と固有ベクトルを取得
	#print(eig_val)
	#print(eig_vec)
	
	return eig_vec
	

#文脈の作成
def make_context(sem_mat, word_list, contex_word, data):#意味行列に文脈を指定する単語リストを入力
	global e
	contex_mat = []
	count = 0
	
	#相関行列(データ行列では列が画像行が単語なので代わりに相関行列を利用)
	relation_mat = np.dot(np.array(data).T, np.array(data))
	
	for word in contex_word:#文脈として選んだ言葉のみ抽出
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
	contex_mat = [0] * len(word_list)
	
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
def sem_projection(sem_mat, sem_contex, data, input_img, contex_vec_list):#dataはデータベースにある画像、input_imgは今回のメイン画像	
	input_vec = np.matrix(input_img) * np.matrix(sem_contex).T
	data_vec = np.matrix(data) * np.matrix(sem_contex).T
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
	
	for f in sem_contex:
		w = 0
		for c in contex_vec_list:
			w += np.dot(np.array(c), np.array(f))
			#print(w)
		weigth_c.append(w / np.linalg.norm(div_vec))
	
	#np.array(input_vec)
	#print(input_vec)
	#距離の計算
	#各（文脈から選抜した）意味素において重みを与えて計算する
	for d in data_vec.tolist():
		count = 0
		tmp = np.array(d) - np.array(input_vec.reshape(-1,))
		tmp = tmp.reshape(-1,)#なんか二次元配列になっちゃう問題
		dis = 0
		
		for w in weigth_c:
			#print(tmp)
			#print(w)
			tmp[count] *= w
			count += 1
			
		for t in tmp:
			dis += t * t
		dis = math.sqrt(dis)
			
		data_dis.append(dis)
	
	#for d in data_vec:
	#	dis = np.linalg.norm(d - input_vec)
	#	data_dis.append(dis)
		#print(str(count) + " : " + str(dis))
		#count += 1
		
	return data_dis
	
	
if __name__ == '__main__':
	data = [[1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 0, 0]]#模範データ(今の段階ではデータベースの既存データとしても利用中)
	word_list = ["sharp", "oval", "pull", "push"]
	contex_word = ["sharp", "oval", "pull"]
	
	sem_mat = make_semantic_matrix(data)
	sem_contex, contex_vec_list = make_context(sem_mat, word_list, contex_word, data)
	
	input_img = [1 ,0 ,0 ,1]#テスト用の入力画像
	data_dis = sem_projection(sem_mat, sem_contex, data, input_img, contex_vec_list)
	print(data_dis)
	