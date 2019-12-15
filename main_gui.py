#codeing:utf-8
# -*- coding: utf-8 -*-

#
#GUIの設定（一旦Tkinterを使っている）
#
#
import sys
import tkinter as tk
from tkinter import ttk
import csv
import glob

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

context_dist_path = "D:\\Sotsuken\\Sotsuken_repo\\result\\output_file\\context_dist\\*"


def write_graph(G):
	global fig, a, canvas, context, graph_list
	
	pos = nx.spring_layout(graph_list[0])
	edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}
	edge_width = [ d['weight']*0.2 for (u,v,d) in G.edges(data=True)]
	
	nx.draw_networkx_nodes(G, pos, with_labels=True,alpha=0.5)
	nx.draw_networkx_labels(G, pos, fontsize=14, font_family="Yu Gothic", font_weight="bold")
	nx.draw_networkx_edges(G, pos,alpha=0.4, edge_color='R', width=edge_width)
	
	#plt.show()
	toolbar = NavigationToolbar2Tk(canvas, graph_frame)
	canvas.get_tk_widget().pack()
	

def apply_context():
	global fig, a, canvas, context, G, graph_list
	
	if(context.get() == "All"):
		G = graph_list[0]
		
	elif(context.get() == "incision"):
		G = graph_list[1]
	
	elif(context.get() == "contusion"):
		G = graph_list[2]
	
	
	write_graph(G)

root = tk.Tk()

root.title(u"Wound Similarity Simmlation System")
root.geometry("1000x500")


#フレーム
graph_frame = tk.Frame(root, relief="ridge")
graph_frame.pack(fill="x", side="left", anchor="nw", padx=2)
info_frame = tk.Frame(root,relief="ridge")
info_frame.pack(fill="x", side="left", anchor="nw", padx=2)
cal_frame = tk.LabelFrame(info_frame,bd=2, relief="ridge", text="Cal. Option")
cal_frame.pack()
detail_frame = tk.LabelFrame(info_frame,bd=2, relief="ridge", text="Detail Option")
detail_frame.pack(fill="x", side="left", anchor="nw", padx=2)


#計算オプション
context = tk.StringVar()
context_combo = ttk.Combobox(cal_frame, state='readonly', textvariable=context)
context_list = ["All", "incision", "contusion"]
context_combo["values"] = tuple(context_list)
context_combo.current(0)
context_combo.pack(fill="x", side="left")


#ボタン
apply_btn = tk.Button(cal_frame,text="適用" ,command=apply_context, width=3, height=1)
apply_btn.pack(fill="x")



#
#グラフの作成
#

#csvの読み込み
folder = context_dist_path
files = glob.glob(folder)

contexts = []#各文脈ごとの距離が書かれている
for file in files:
	csv_file = open(file, "r", encoding="utf_8", errors="", newline="")
	f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
	
	distance = []
	for row in f:
		if(len(row) == 0):
			continue
		distance.append(row)
	
	contexts.append(np.array(distance))
	
#print(contexts)

#文脈ごとにグラフを作成
graph_list = []
for c in contexts:
	G = nx.Graph()
	num = len(c)
	for i in range(1,num + 1):
		G.add_node(i)
	
	for i in range(1, num):
		for j in range(i + 1, num + 1):
			w = float(c[i - 1][j - 1].replace('"',''))
			G.add_edge(i, j, weight=w*10)
	
	graph_list.append(G)
	


#グラフ表示
fig = Figure()
a = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=graph_frame)


#G = nx.karate_club_graph()
write_graph(graph_list[0])


root.mainloop()