#ocrプログラム

#①ライブラリのインポート
import numpy as np    
import neurolab as nl
from sklearn.cluster import KMeans
import os; os.environ['OMP_NUM_THREADS'] = '3'

#②変数定義(use_n：読み込むデータ数 train_n：学習に使用するデータ数)
ans_labels="abcdefghijklmnopqrstuvwxyz"
use_n = 50000
train_n = int(0.90*use_n)
pixel_n = 8*16
ans_n = len(ans_labels)
data = []
labels = []

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

#④データ形式の変換（labelsとdataをリスト→配列）            
labels = np.array(labels).reshape(-1, ans_n)
data = np.array(data).reshape(-1, pixel_n)# 大きさは50000

# 各文字に対応するデータを格納するための辞書を初期化
average_data = {char: [] for char in ans_labels}# 各文字がキーで、それぞれ2つの値を持つ

# データとラベルをループして、各文字に対応するデータを辞書に追加
for d, l in zip(data[:train_n], labels[:train_n]):
    char = ans_labels[np.argmax(l)]
    average_data[char].append(d)  # average_data['a'].append(data)みたいになる
                                  # この時点では辞書の大きさは45000

for char in average_data:
    # average_data[char]をnumpy配列に変換します。
    data_abc = np.array(average_data[char])

    # KMeansクラスを初期化します。ここでは、クラスタ数を2に設定し、初期化方法を'k-means++'に設定しています。
    kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=0)

    # データに対してk-meansアルゴリズムを適用します。
    kmeans.fit(data_abc)

    # クラスタ中心をaverage_data[char]に格納します。
    average_data[char] = kmeans.cluster_centers_


# print(average_data['a'])

# 学習素材でテスト
correct = 0
test_n = use_n - train_n  # テストデータの数を更新
# 推測方法
for i in range(test_n): 
    min_diff = float('inf')  # 最小の差分を保存するための変数を初期化
    predicted_label = None  # 予測ラベルを保存するための変数を初期化
    for j in range(26):  # 各代表的な画像に対してループ
        char = ans_labels[j]
        for k in range(len(average_data[char])):  # 各文字に対して複数の画像があるため、それぞれに対してループ
            sum_diff = 0  # 各テスト画像ごとにsum_diffをリセット
            for d in range(len(data[train_n+i])):  # data[]はどれも大きさが128なので、128回ループ
                sum_diff += abs(average_data[char][k][d] - data[train_n+i][d])
            if sum_diff < min_diff:  # もし新しい差分が最小の差分よりも小さい場合
                min_diff = sum_diff  # 最小の差分を更新
                predicted_label = j  # 予測ラベルを更新

    # 正解ラベルを取得
    true_label = np.argmax(labels[train_n+i])

    # 正解数を更新
    if predicted_label == true_label:
        correct += 1

# 精度を計算する
print(f"結果: {correct / test_n}")

import matplotlib.pyplot as plt
from PIL import Image

# average_data内の各文字に対してループ
for char in average_data:
    # 各文字に対して複数の画像データがあるため、それぞれに対してループ
    for i in range(len(average_data[char])):
        # 画像データを取得し、16x8の形状に変形
        img_data = average_data[char][i].reshape((16, 8))
        
        # データを0-255の整数に変換（黒が1、白が0）
        img_data = np.uint8((1.0 - img_data) * 255)
        
        # PIL.Imageオブジェクトを作成
        img = Image.fromarray(img_data)
        
        # 画像をPNGファイルとして保存。ファイル名には文字とインデックスを含めます。
        img.save(f"kmeans_{len(average_data[char])}_{char}_{i}.png")


#print(sum_diff)



"""
test:5000
クラスタ数: 2 -> 結果: 0.4096
クラスタ数: 3 -> 結果: 0.4246
クラスタ数: 4 -> 結果: 0.4788
クラスタ数: 5 -> 結果: 0.475



    https://nisshingeppo.com/ai/whats-k-means/
    を参照。

import matplotlib.pyplot as plt
from PIL import Image

# average_data内の各文字に対してループ
for char in average_data:
    # 各文字に対して複数の画像データがあるため、それぞれに対してループ
    for i in range(len(average_data[char])):
        # 画像データを取得し、16x8の形状に変形
        img_data = average_data[char][i].reshape((16, 8))
        
        # データを0-255の整数に変換（黒が1、白が0）
        img_data = np.uint8((1.0 - img_data) * 255)
        
        # PIL.Imageオブジェクトを作成
        img = Image.fromarray(img_data)
        
        # 画像をPNGファイルとして保存。ファイル名には文字とインデックスを含めます。
        img.save(f"{kmeans}_{char}_{i}.png")

"""

