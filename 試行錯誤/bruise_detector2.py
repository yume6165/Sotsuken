import os, sys, time
import tkinter as tk
from tkinter import messagebox as tkMessageBox
from tkinter import filedialog as tkFileDialog
from tkinter import ttk
import cv2 as cv
from PIL import Image, ImageTk

win_width = 1130#ウィンドウサイズ
win_hight = 640
can_width = 550#canvasサイズ
can_hight = 550

c_center_x = can_width / 2 + 1#距離を示す円の中心（初期値）
c_center_y = can_hight / 2 + 1

l_oval_r = (can_width - 4) / 2#最大円の半径
r_oval_r = (can_width - 4) / 4#中大円の半径
s_oval_r = (can_width - 4) / 16#最小円の半径

pressed_x = 0
pressed_y = 0
item_id = -1

global img_count_pro11, img_count_pro12, img_count_pro13, img_count_pro14
global img_count_pro21, img_count_pro22, img_count_pro23, img_count_pro24
img_count_pro11 = 0
img_count_pro12 = 0
img_count_pro13 = 0
img_count_pro14 = 0
img_count_pro21 = 0
img_count_pro22 = 0
img_count_pro23 = 0
img_count_pro24 = 0
img_count = 0
filter_flag = 0
blevel_frag1 = 0#画像列１の2値化を行ったかどうかのフラグ
blevel_frag2 = 0#画像列２の2値化を行ったかどうかのフラグ

def rewrite(path):#画像の書き換え
	global image_pil, image_bgr, image_rgb, image_tk, image_bgr_re#画像を表示するために一時的に画像データを入れる変数
	image_bgr = cv.imread(path)
	height = image_bgr.shape[0]
	width = image_bgr.shape[1]
	image_bgr_re = cv.resize(image_bgr,(190, int(int(height)*190/int(width))))
	image_rgb = cv.cvtColor(image_bgr_re, cv.COLOR_BGR2RGB)
	image_pil = Image.fromarray(image_rgb)#PILフォーマットへ
	image_tk = ImageTk.PhotoImage(image_pil)
	

def blevel():#N値化の関数
	global image_pil, image_bgr, image_rgb, image_tk, image_bgr_re#画像を表示するために一時的に画像データを入れる変数
	global img_pro11, img_pro12, img_pro13,img_pro14
	global img_pro21, img_pro22, img_pro23,img_pro24
	global img_count_pro11, img_count_pro12, img_count_pro13, img_count_pro14
	global img_count_pro21, img_count_pro22, img_count_pro23, img_count_pro24
	global thre, blevel_count1, blevel_frag1
	
	if(radio1_value.get() == 0 and radio2_value.get() == 0):#画像処理の系列を選んでいない場合は何もしない
		print("何もしません")
		return 0
		
	elif(radio1_value.get() == 1):#画像処理列１が選ばれたとき	
		threshold = int(thre.get())
		if(blevel_frag1 == 0):#初めて2値化処理をしたとき
			blevel_count1 = img_count_pro11 - 1
			blevel_frag1 = 1
			
		print("dhcuw")
		image_bgr = cv.imread("D:\Sotsuken\output\pro11\pro11_"+ str(blevel_count1)+".jpg",0)
		ret, image_bgr = cv.threshold(image_bgr, threshold, 255, cv.THRESH_BINARY)
		cv.imwrite("D:\Sotsuken\output\pro11\pro11_"+ str(img_count_pro11)+".jpg", image_bgr)
		img_count_pro11 += 1
		rewrite("D:\Sotsuken\output\pro11\pro11_"+ str(img_count_pro11 - 1)+".jpg")
		img_pro11 = image_tk
		canvas10.create_image(0, 0, image=img_pro11, anchor='nw', tag="pro11")
			
		
	
	elif(radio2_value.get() == 1):#画像処理列2が選ばれたとき	
		return 0
		
	

def filter_option():#フィルタごとにオプションを表示j
	global filter #フィルターの名前が入ってるよ
	global frame9
	global domain, domain_box, sigmaC, sigmaC_box, sigmaS, sigmaS_box#バイラテラルフィルタの変数
	global domain_label, sigmaC_label, sigmaS_label#バイラテラルフィルタの変数
	global filter_flag
	
	if(filter.get() == "平滑化フィルタ"):
		filter_flag = 1
		frame9.destroy()
		frame9 = tk.LabelFrame(frame8, relief="ridge", padx=2, pady=3, text="Option")
		frame9.pack(fill="x")
		button6 = tk.Button(frame9, text="適用", command=adfilter)
		button6.pack(fill="x")
		return 0
		
	elif(filter.get() == "バイラテラルフィルタ"):
		if(filter_flag == 1):#二回目以上の呼び出し
			frame9.destroy()
			
		frame9 = tk.LabelFrame(frame8, relief="ridge", padx=2, pady=3, text="Option")
		frame9.pack(fill="x")
		
		domain_label = tk.Label(frame9, text="領域")
		domain_label.grid(row=0, column=0, padx=3)
		sigmaC_label = tk.Label(frame9, text="sigmaColor")
		sigmaC_label.grid(row=0, column=1, padx=3)
		sigmaS_label = tk.Label(frame9, text="sigmaSolor")
		sigmaS_label.grid(row=0, column=2, padx=3)
		
		domain = tk.StringVar()
		domain.set("15")
		domain_box = tk.Entry(frame9, textvariable=domain, width=10)
		domain_box.grid(row=1, column=0, padx=3)
		sigmaC = tk.StringVar()
		sigmaC.set("20")
		sigmaC_box = tk.Entry(frame9, textvariable=sigmaC, width=10)
		sigmaC_box.grid(row=1, column=1, padx=3)
		sigmaS = tk.StringVar()
		sigmaS.set("20")
		sigmaS_box = tk.Entry(frame9, textvariable=sigmaS, width=10)
		sigmaS_box.grid(row=1, column=2, padx=3)
		
		button6 = tk.Button(frame9, text="適用", command=adfilter)
		button6.grid(row=1, column=3, padx=3)
		return 0
		
		
	
def adfilter():#フィルタの適用関数
	global filter #フィルターの名前が入ってるよ
	global image_pil, image_bgr, image_rgb, image_tk, image_bgr_re#画像を表示するために一時的に画像データを入れる変数
	global img_pro11, img_pro12, img_pro13,img_pro14
	global img_pro21, img_pro22, img_pro23,img_pro24
	global img_count_pro11, img_count_pro12, img_count_pro13, img_count_pro14
	global img_count_pro21, img_count_pro22, img_count_pro23, img_count_pro24
	global domain, sigmaC, sigmaS#バイラテラルフィルタの変数
	
	if(radio1_value.get() == 0 and radio2_value.get() == 0):#画像処理の系列を選んでいない場合は何もしない
		print("何もしません")
		return 0
		
	elif(radio1_value.get() == 1):#画像処理列１が選ばれたとき	
		if(filter.get() == "バイラテラルフィルタ"):#バイラテラルフィルタの処理をする
			print("dhcuw")
			image_bgr = cv.imread("D:\Sotsuken\output\pro11\pro11_"+ str(img_count_pro11 - 1)+".jpg")
			image_bgr = cv.bilateralFilter(image_bgr, int(domain.get()), int(sigmaC.get()), int(sigmaS.get()))
			cv.imwrite("D:\Sotsuken\output\pro11\pro11_"+ str(img_count_pro11)+".jpg", image_bgr)
			img_count_pro11 += 1
			rewrite("D:\Sotsuken\output\pro11\pro11_"+ str(img_count_pro11 - 1)+".jpg")
			img_pro11 = image_tk
			canvas10.create_image(0, 0, image=img_pro11, anchor='nw', tag="pro11")
	
	elif(radio2_value.get() == 1):#画像処理列１が選ばれたとき
		if(filter.get() == "バイラテラルフィルタ"):#バイラテラルフィルタの処理をする
			image_bgr = cv.imread("D:\Sotsuken\output\pro21\pro21_"+ str(img_count_pro21 - 1)+".jpg")
			image_bgr = cv.bilateralFilter(image_bgr, int(domain.get()), int(sigmaC.get()), int(sigmaS.get()))
			cv.imwrite("D:\Sotsuken\output\pro21\pro21_"+ str(img_count_pro21)+".jpg", image_bgr)
			img_count_pro21 += 1
			rewrite("D:\Sotsuken\output\pro21\pro21_"+ str(img_count_pro21 - 1)+".jpg")
			img_pro21 = image_tk
			canvas20.create_image(0, 0, image=img_pro21, anchor='nw', tag="pro21")
			
			if(img_count >= 2):#画像を2枚以上読み込んでいたら	
				image_bgr = cv.imread("D:\Sotsuken\output\pro22\pro22_"+ str(img_count_pro22 - 1)+".jpg")
				image_bgr = cv.bilateralFilter(image_bgr, int(domain.get()), int(sigmaC.get()), int(sigmaS.get()))
				cv.imwrite("D:\Sotsuken\output\pro22\pro22_"+ str(img_count_pro22)+".jpg", image_bgr)
				img_count_pro22 += 1
				rewrite("D:\Sotsuken\output\pro22\pro22_"+ str(img_count_pro22 - 1)+".jpg")
				img_pro22 = image_tk
				canvas21.create_image(0, 0, image=img_pro22, anchor='nw', tag="pro22")
			
	return 0


def togray():
	global image_pil, image_bgr, image_rgb, image_tk, image_bgr_re#画像を表示するために一時的に画像データを入れる変数
	global img_pro11, img_pro12, img_pro13,img_pro14
	global img_pro21, img_pro22, img_pro23,img_pro24
	global img_count_pro11, img_count_pro12, img_count_pro13, img_count_pro14
	global img_count_pro21, img_count_pro22, img_count_pro23, img_count_pro24
	global togay_tmp1, togay_tmp2, togay_tmp3, togay_tmp4#グレースケールのための画像補完
	if(radio1_value.get() == 0 and radio2_value.get() == 0):#画像処理の系列を選んでいない場合は何もしない
		print("何もしません")
		return 0
	else:
		if (radio1_value.get() == 1):#画像処理列１が選ばれたとき
			togay_tmp1 = cv.imread("D:\Sotsuken\output\pro11\pro11_"+ str(img_count_pro11 - 1)+".jpg")
			togay_tmp1 = cv.cvtColor(togay_tmp1, cv.COLOR_BGR2GRAY)
			cv.imwrite("D:\Sotsuken\output\pro11\pro11_"+ str(img_count_pro11)+".jpg", togay_tmp1)
			img_count_pro11 += 1
			rewrite("D:\Sotsuken\output\pro11\pro11_"+ str(img_count_pro11 - 1)+".jpg")
			img_pro11 = image_tk
			canvas10.create_image(0, 0, image=img_pro11, anchor='nw', tag="pro11")
			
			
			if(img_count >= 2):#画像を2枚以上読み込んでいたら
				togay_tmp2 = cv.imread("D:\Sotsuken\output\pro12\pro12_"+ str(img_count_pro12 - 1)+".jpg")
				togay_tmp2 = cv.cvtColor(togay_tmp2, cv.COLOR_BGR2GRAY)
				cv.imwrite("D:\Sotsuken\output\pro12\pro12_"+ str(img_count_pro12)+".jpg", togay_tmp2)
				img_count_pro12 += 1
				rewrite("D:\Sotsuken\output\pro12\pro12_"+ str(img_count_pro12 - 1)+".jpg")
				img_pro12 = image_tk
				canvas11.create_image(0, 0, image=img_pro12, anchor='nw', tag="pro12")

			else:
				return 0
		
		else:
			togay_tmp1 = cv.imread("D:\Sotsuken\output\pro21\pro21_"+ str(img_count_pro21 - 1)+".jpg")
			togay_tmp1 = cv.cvtColor(togay_tmp1, cv.COLOR_BGR2GRAY)
			cv.imwrite("D:\Sotsuken\output\pro21\pro21_"+ str(img_count_pro21)+".jpg", togay_tmp1)
			img_count_pro21 += 1
			rewrite("D:\Sotsuken\output\pro21\pro21_"+ str(img_count_pro21 - 1)+".jpg")
			img_pro21 = image_tk
			canvas20.create_image(0, 0, image=img_pro21, anchor='nw', tag="pro21")
				
			if(img_count >= 2):
				togay_tmp2 = cv.imread("D:\Sotsuken\output\pro22\pro22_"+ str(img_count_pro22 - 1)+".jpg")
				togay_tmp2 = cv.cvtColor(togay_tmp2, cv.COLOR_BGR2GRAY)
				cv.imwrite("D:\Sotsuken\output\pro22\pro22_"+ str(img_count_pro22)+".jpg", togay_tmp2)
				img_count_pro22 += 1
				rewrite("D:\Sotsuken\output\pro22\pro22_"+ str(img_count_pro22 - 1)+".jpg")
				img_pro22 = image_tk
				canvas21.create_image(0, 0, image=img_pro22, anchor='nw', tag="pro22")
			else:
				return 0
			

def radio1_clicked():#加工する画像列の１番目が選ばれたとき
	radio1_value.set(1)
	radio2_value.set(0)
	global img_pro11, img_pro12#真ん中の列の画像を表示するデータを入れる
	
	return 0
	
def radio2_clicked():#加工する画像列の２番目が選ばれたとき
	radio2_value.set(1)
	radio1_value.set(0)
	global img_pro21, img_pro22#真ん中の列の画像を表示するデータを入れる
	return 0


def pressed(event):
	print("おされたー")
	global pressed_x, pressed_y, item_id
	item_id = canvas.find_closest(event.x, event.y)
	tags = canvas.gettags(item_id[0])
	
	for tag in tags:
		delta_x = event.x - pressed_x
		delta_y = event.y - pressed_y
		
	pressed_x = event.x
	pressed_y = event.y


def dragged(event):#canvas内のエリアをクリックしたときの操作
	print("動かされたー")
	global pressed_x, pressed_y, item_id
	item_id = canvas.find_closest(event.x, event.y)
	tags = canvas.gettags(item_id[0])
	
	for tag in tags:
		delta_x = event.x - pressed_x
		delta_y = event.y - pressed_y
		
	pressed_x = event.x
	pressed_y = event.y
	
	
def openfile():#ファイルを開くときの操作
	fTyp = [('JPEGファイル', '.jpg')]
	iDir = '/home/ユーザ名/'
	
	#ひとつのファイルを選択する
	filename = tkFileDialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
	return filename
	
	#複数のファイルを選択する
	filenames = filedialog.askopenfilenames(filetypes=fTyp, initialdir=iDir)
	
	for f in filenames:
		return f
		
	dirname = tkFileDialog.askdirectory(initialdir=iDir)
	
	tkMessageBox.showinfo('SELECTIED DIRECTRY is ...', filename)
	
	
	
def button1_clicked():#ファイルの参照ボタンのアクション
	file1.set(openfile())
	#fTyp = [("", "*")]
	#iDir = os.path.abspath(os.path.dirname(__file__))
	#filepath = filedialog.askopenfilename(filetypes = fTyp, initialdir=iDir)
	#file1.set(filepath)


def button2_clicked():#画像表示のアクション
	global image_pil, image_bgr, image_rgb, image_tk, path, image_bgr_re#画像を表示するために一時的に画像データを入れる変数
	global img_ori1, img_ori2, img_ori3, img_ori4, img_count
	global img_count_pro11, img_count_pro12, img_count_pro13, img_count_pro14
	global img_count_pro21, img_count_pro22, img_count_pro23, img_count_pro24
	
	path = file1.get().replace('/', '\\')
	if(path != ""):
		image_bgr = cv.imread(path)
		img_list.append(image_bgr)#bgr読み込んだ画像を画像リストに追加
		cv.imwrite("D:\Sotsuken\output\ori\ori"+str(img_count)+".jpg", image_bgr)
		#読み込んだ画像を縮小（データとして残しているのは縮小前データ）
		height = image_bgr.shape[0]
		width = image_bgr.shape[1]
		image_bgr_re = cv.resize(image_bgr, (190, int(int(height)*190/int(width))))
		image_rgb = cv.cvtColor(image_bgr_re, cv.COLOR_BGR2RGB)
		#image_rgb = cv.cvtColor(image_bgr_re, cv.COLOR_BGR2GRAY)
		image_pil = Image.fromarray(image_rgb)#PILフォーマットへ
		image_tk = ImageTk.PhotoImage(image_pil)
		img_count = len(img_list) - 1
	
		if img_count == 0:
			cv.imwrite("D:\Sotsuken\output\pro11\pro11_"+ str(img_count_pro11)+".jpg", image_bgr)
			cv.imwrite("D:\Sotsuken\output\pro21\pro21_"+ str(img_count_pro21)+".jpg", image_bgr)
			img_count_pro11 += 1
			img_count_pro21 += 1
			#canvas00.delete("def1")
			img_ori1 = image_tk
			img_pro11 = image_tk
			img_pro21 = image_tk
			canvas00.create_image(0, 0, image=img_ori1, anchor='nw', tag="ori1")
			canvas10.create_image(0, 0, image=img_pro11, anchor='nw', tag="pro11")
			canvas20.create_image(0, 0, image=img_pro21, anchor='nw', tag="pro21")
			img_count += 1
		elif img_count == 1:
			cv.imwrite("D:\Sotsuken\output\pro12\pro12_"+ str(img_count_pro12)+".jpg", image_bgr)
			cv.imwrite("D:\Sotsuken\output\pro22\pro22_"+ str(img_count_pro22)+".jpg", image_bgr)
			img_ori2 = image_tk
			img_count_pro12 += 1
			img_count_pro22 += 1
			img_pro12 = image_tk
			img_pro22 = image_tk
			canvas01.create_image(0, 0, image=img_ori2, anchor='nw', tag="ori2")
			canvas11.create_image(0, 0, image=img_pro12, anchor='nw', tag="pro12")
			canvas21.create_image(0, 0, image=img_pro22, anchor='nw', tag="pro22")
			img_count += 1
		
		
		else:
			print("これ以上画像を読み込めません")
	
	else:
		print("画像を選択してください")
		
	
def status_check():#Toolbarの画像の加工状況を更新
	return 0
	

img_list = []#読み込んだオリジナル画像のリスト
img_list_pro11 = []#加工する画像のリスト１左
img_list_pro12 = []#加工する画像のリスト１左から2番目
img_list_pro21 = []#加工する画像のリスト2左
img_list_pro22 = []#加工する画像のリスト2左から2番目



root = tk.Tk()
root.title("GUI test")
root.geometry(str(win_width)+"x"+str(win_hight))
	
frame1 = tk.LabelFrame(root, bd=2, relief="ridge", text="Tool Bar")
frame1.pack(fill="y", side="left", anchor="nw", padx=2)

frame2 = tk.LabelFrame(root, bd=2, relief="ridge", text="Image")
frame2.pack(fill="both",anchor="nw", padx=2, ipadx=2, ipady=2)
	
frame4 = tk.LabelFrame(frame1, relief="ridge",text="Input Image")
frame4.pack(fill="x")

#参照ボタンの作成
button1 = tk.Button(frame4, text=u'参照', command=button1_clicked)
button1.grid(row=1, column=1)
	
#参照ファイルパス表示ラベルの作成
file1 = tk.StringVar()
file1_entry = tk.Entry(frame4, textvariable=file1, width=30)
file1_entry.grid(row=1, column=0, padx=2)
	
#読み込みボタンの作成
button2 = tk.Button(frame4, text='読み込み', command=button2_clicked)
button2.grid(row=1, column=2, padx=2)

#変更する画像列を選択
frame6 = tk.LabelFrame(frame1, relief="ridge", padx=2, pady=3, text="Select Edit Image")
frame6.pack(fill="x")
radio1_value = tk.IntVar()
radio2_value = tk.IntVar()
radio1_value.set(0)
radio2_value.set(0)
radio1 = ttk.Radiobutton(frame6, value=radio1_value, variable=1, text='P Image 1', command=radio1_clicked)
radio1.state(['selected'])
radio1.pack(side="left", fill="x")
radio2 = ttk.Radiobutton(frame6, value=radio2_value, variable=1, text='P Image 2', command=radio2_clicked)
radio2.pack(side="right", fill="x")

#グレースケールに変換
frame7 = tk.Frame(frame1,  relief="ridge", padx=2, pady=3)
frame7.pack(fill="x")
button4 = tk.Button(frame7, text='グレースケール', command=togray, width=30, height=1)
button4.pack(fill="x")

#2値化
frame11 = tk.LabelFrame(frame1,  relief="ridge", padx=2, pady=3, text="N値化")
frame11.pack(fill="x")
thre = tk.StringVar()
thre.set("100")
thre_box = tk.Entry(frame11, textvariable=thre, width=20)
thre_box.pack(side="left", fill="x")
button7 = tk.Button(frame11, text="適用", command=blevel)
button7.pack(side="right", fill="x")


#フィルタの適用
frame8 = tk.LabelFrame(frame1, relief="ridge", padx=2, pady=3, text="Filter")
frame8.pack(fill="x")
frame10 = tk.Frame(frame8, relief="ridge", padx=2, pady=3)
frame10.pack(fill="x")
filter = tk.StringVar()
filter_combo = ttk.Combobox(frame10, state='readonly', textvariable=filter)
filter_list = ["平滑化フィルタ","バイラテラルフィルタ"]
filter_combo["values"] = tuple(filter_list)
filter_combo.current(0)
filter_combo.pack(fill="x", side="left")
button5 = tk.Button(frame10, text="オプション表示", command=filter_option)
button5.pack(side="right")


#適用ボタン
#frame5 = tk.Frame(frame1, relief="ridge", padx=2, pady=3)
#frame5.pack()
#button3 = tk.Button(frame5, text='適用', command=addapt, width=30, height=1)
#button3.pack()




#画像の表示領域
#オリジナル画像の表示領域
frame3 = tk.LabelFrame(frame2, bd=2, relief="ridge", text="Original Image")
frame3.pack(fill="x")
canvas00 = tk.Canvas(frame3, width=200, height=190) # Canvas作成
canvas00.create_oval(10,10,100,100, tag="def1")
canvas00.pack(fill="x", side="left")
canvas01 = tk.Canvas(frame3, width=200, height=190) # Canvas作成
canvas01.create_oval(10,10,100,100, tag="def2")
canvas01.pack(fill="x", side="left")



#加工画像１の表示領域
frame4 = tk.LabelFrame(frame2, bd=2, relief="ridge", text="Proccessed Image 1")
frame4.pack(fill="x")
canvas10 = tk.Canvas(frame4, width=200, height=190) # Canvas作成
canvas10.create_oval(10,10,100,100, tag="cha11")
canvas10.pack(fill="x", side="left")
canvas11 = tk.Canvas(frame4, width=200, height=190) # Canvas作成
canvas11.create_oval(10,10,100,100, tag="cha12")
canvas11.pack(fill="x", side="left")




#canvas.create_oval(c_center_x - l_oval_r, c_center_y - l_oval_r, c_center_x + l_oval_r, c_center_y + l_oval_r, fill="yellow", outline="yellow", tags="l_oval")
#canvas.create_oval(c_center_x - r_oval_r, c_center_y - r_oval_r, c_center_x + r_oval_r, c_center_y + r_oval_r, fill="#ffd700", outline="#ffd700", tags="r_oval")
#canvas.create_oval(c_center_x - s_oval_r, c_center_y - s_oval_r, c_center_x + s_oval_r, c_center_y + s_oval_r, fill="#ffa500", outline="#ffa500", tags="s_oval")
#canvas.place(x=300, y=10)
	
#イベントを追加
#canvas.tag_bind("rect", "<1>", pressed)#名前に＜＞を付けないとちゃんと動かない
#canvas.tag_bind("l_oval", "<1>", pressed)
#canvas.tag_bind("r_oval", "<1>", pressed)
#canvas.tag_bind("s_oval", "<1>", pressed)
#canvas.tag_bind("rect", "<1>", dragged)#名前に＜＞を付けないとちゃんと動かない
#canvas.tag_bind("l_oval", "<1>", dragged)
#canvas.tag_bind("r_oval", "<1>", dragged)
#canvas.tag_bind("s_oval", "<1>", dragged)
	
	
	#ここにレイアウトを書いていきます
	
	
root.mainloop()