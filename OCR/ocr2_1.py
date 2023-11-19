#ocrプログラム

#①ライブラリのインポート
import numpy as np    
import neurolab as nl

#②変数定義(use_n：読み込むデータ数 train_n：学習に使用するデータ数)
ans_labels="abcdefghijklmnopqrstuvwxyz"
use_n = 50000
train_n = int(0.3*use_n)
pixel_n = 8*16
ans_n = len(ans_labels)
data = []
labels = []

#③学習素材の読み込み
with open('letter_1.data', 'r') as f:
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

#④データ形式の変換（labelsとdataをリスト→配列）            
labels = np.array(labels).reshape(-1, ans_n)
data = np.array(data).reshape(-1, pixel_n)# 大きさは50000

# 各文字に対応するデータを格納するための辞書を初期化
average_data = {char: [] for char in ans_labels}

# データとラベルをループして、各文字に対応するデータを辞書に追加
for d, l in zip(data[:26], labels[:26]):# 26まで（[25]まで）ループする
    char = ans_labels[np.argmax(l)]
    average_data[char].append(d)  # average_data['a'].append(data)みたいになる

# 各文字に対応するデータの平均を計算
for char in average_data:
    average_data[char] = np.mean(average_data[char], axis=0)

print(average_data)

# 学習素材でテスト
correct = 0
test_n = int(use_n-train_n)
# 推測方法
for i in range(test_n): 
    sum_diff = np.zeros(26)  # 各テスト画像ごとにsum_diffをリセット
    for j in range(26):  # 各代表的な画像に対してループ
        char = ans_labels[j]
        for d in range(len(data[train_n+i])):# data[]はどれも大きさが128なので、128回ループ
            sum_diff[j] += abs(average_data[char][d] - data[train_n+i][d])

    # 最小のsum_diffを持つ文字を見つける
    predicted_label = np.argmin(sum_diff)

    # 正解ラベルを取得
    true_label = np.argmax(labels[train_n+i])

    # 正解数を更新
    if predicted_label == true_label:
        correct += 1

# 精度を計算する
print(f"結果: {correct / test_n}")

#print(sum_diff)

import matplotlib.pyplot as plt
from PIL import Image

# average_data内の各文字に対してループ ループする回数はaverage_dataの大きさ分だけど、charがaやbに対応している
for char in average_data:
    # 画像データを取得し、16x8の形状に変形
    img_data = average_data[char].reshape((16, 8))
    
    # データを0-255の整数に変換（黒が1、白が0）
    img_data = np.uint8((1.0 - img_data) * 255)
    
    # PIL.Imageオブジェクトを作成
    img = Image.fromarray(img_data)
    
    # 画像をPNGファイルとして保存
    img.save(f"{char}.png")


"""
0.90 -> 結果: 0.3152(test:5000)
0.80 -> 結果: 0.3142(test:10000)
0.60 -> 結果: 0.3102(test:20000)
0.30 -> 結果: 0.3105142857142857(test:35000)

"""