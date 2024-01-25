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

def ocr3_program(filename):
    with Image.open(filename) as img:
        
        # 画像をグレースケールに変換
        img_gray = img.convert('L')
        # 画像のデータをNumpy配列に変換
        img_array = np.array(img_gray)
        # ピクセルの値を0～1の範囲に正規化
        img_normalized = 1 - img_array / 255.0

        # 画像データを1次元配列に変換
        img_normalized = img_normalized.ravel()

        # 保存した辞書を読み込む
        with open('average_data_k.pkl', 'rb') as f:
            average_data = pickle.load(f)

        # 推測方法
        min_diff = float('inf')  # 最小の差分を保存するための変数を初期化
        predicted_label = None  # 予測ラベルを保存するための変数を初期化
        for j in range(26):  # 各代表的な画像に対してループ
            char = ans_labels[j]
            for k in range(len(average_data[char])):  # 各文字に対して複数の画像があるため、それぞれに対してループ
                sum_diff = 0  # 各テスト画像ごとにsum_diffをリセット
                for d in range(128):  # data[]はどれも大きさが128なので、128回ループ
                    sum_diff += abs(average_data[char][k][d] - img_normalized[d])
                if sum_diff < min_diff:  # もし新しい差分が最小の差分よりも小さい場合
                    min_diff = sum_diff  # 最小の差分を更新
                    predicted_label = j  # 予測ラベルを更新
        
        predicted_label = ans_labels[predicted_label]
        return predicted_label


