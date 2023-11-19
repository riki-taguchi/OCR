#ocrプログラム

#①ライブラリのインポート
import numpy as np    
import neurolab as nl

#②変数定義(use_n：読み込むデータ数 train_n：学習に使用するデータ数)
ans_labels="abcdefghijklmnopqrstuvwxyz"
use_n = 9
train_n = int(0.9*use_n)
pixel_n = 8*16
ans_n = len(ans_labels)
data = []
target = 'f'

#③学習素材の読み込み
with open('letter.data', 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        
        if list_vals[1] == target:
            #手書きデータ（入力データ）
            char = np.array([float(x) for x in list_vals[6:-1]])
            data.append(char)

            if len(data) >= use_n:
                break

        

#④データ形式の変換（labelsとdataをリスト→配列 どっちも二次元配列）            
data = np.array(data).reshape(-1, pixel_n)

from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

# average_data内の各文字に対してループ
for char in range(use_n):
    # 画像データを取得し、16x8の形状に変形
    img_data = data[char].reshape((16, 8))
    
    # データを0-255の整数に変換（黒が1、白が0）
    img_data = np.uint8((1.0 - img_data) * 255)
    
    # PIL.Imageオブジェクトを作成
    img = Image.fromarray(img_data)
    
    # 現在の日時を取得
    now = datetime.now()

    # 日時を指定の形式で出力
    datetime_str = now.strftime("%Y%m%d%H%M")

    # 画像をPNGファイルとして保存
    img.save(f"new_{target}_{datetime_str}_{char+1}.png")


"""
#③学習素材の読み込み
with open('letter.data', 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        
        #正解ラベル（出力データ）
        label = np.zeros((ans_n, 1))
        label[ans_labels.index(list_vals[1])] = 1 
        labels.append(label)
        
        #手書きデータ（入力データ）
        char = np.array([float(x) for x in list_vals[6:-1]])
        data.append(char)

        if len(data) >= use_n:
            break

"""
