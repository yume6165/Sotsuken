#コンソールからファイルのディレクトリを受け取って，ファイル（文字列、、よりはあパスのほうがいい？）のリストを返す

import os, sys, time
import glob

dir = sys.stdin.readline()#目的のディレクトリまでのパス
dir = dir[:-1]
#dir = 'D:\\Sotsuken\\webapp\\public\\input\\*'

if __name__ == '__main__':
	file_list = glob.glob(dir)
	print(file_list)