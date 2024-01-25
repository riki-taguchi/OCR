#ocrプログラム
""" 
画像ファイルを10のNumpy配列にする。それを、辞書の画像とどれだけ類似しているか計算する。
一番類似していたアルファベットであると特定する。
"""
#①ライブラリのインポート
import numpy as np    
import neurolab as nl
import pickle

from PIL import Image
import numpy as np

ans_labels="abcdefghijklmnopqrstuvwxyz"

def ocr4_program(filename):
    with Image.open(filename) as img:
        
        # 画像をグレースケールに変換
        img_gray = img.convert('L')
        # 画像のデータをNumpy配列に変換
        img_array = np.array(img_gray)
        # ピクセルの値を0～1の範囲に正規化
        img_normalized = 1 - img_array / 255.0

        # 画像データを1次元配列に変換し、その後2次元配列に変換
        img_normalized = img_normalized.ravel().reshape(1, -1)

        # モデルと辞書を読み込む
        with open('model.pkl', 'rb') as f:
            gmm = pickle.load(f)

        with open('dict.pkl', 'rb') as f:
            cluster_names = pickle.load(f)

        # 推測方法
            
        predicted_labels_position = gmm.predict(img_normalized)
        predicted_label = cluster_names[predicted_labels_position[0]]

        return predicted_label

