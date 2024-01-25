# ライブラリのインポート
import numpy as np    
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

# 変数定義(use_n：読み込むデータ数 train_n：学学習に使用するデータ数)
ans_labels="abcdefghijklmnopqrstuvwxyz"
use_n = 50000
train_n = int(0.90*use_n)
pixel_n = 8*16
ans_n = len(ans_labels)
data = []
labels = []
labels_string_int = []

# 学学習素材の読み込み with open('letter.data', 'r') as f:, with open('/content/drive/My Drive/OCR/letter.data', 'r') as f:
with open('letter.data', 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        
        #正解ラベル（出力データ）
        label = np.zeros((ans_n, 1))
        label[ans_labels.index(list_vals[1])] = 1 
        labels.append(label)
        labels_string_int.append(ans_labels.index(list_vals[1]))

        #手書きデータ（入力データ）
        char = np.array([float(x) for x in list_vals[6:-1]])
        data.append(char)

        if len(data) >= use_n:
            break

# データ形式の変換（labelsとdataをリスト→配列）            
labels = np.array(labels).reshape(-1, ans_n)
data = np.array(data).reshape(-1, pixel_n)# 大きさは50000
labels_string_int = np.array(labels_string_int)
train_data = data[:train_n]
test_data = data[train_n:use_n]
train_labels_string_int = labels_string_int[:train_n]
test_labels_string_int = labels_string_int[train_n:use_n]

# 混合ガウス分布の作成（n_components=26, covariance_type='full'）
gmm = mixture.GaussianMixture(n_components=182, covariance_type='full')

# データセットの学習（fitメソッド）
gmm.fit(train_data)

# 26個のクラスタに名前はついていなくて、そのクラスタに分類された画像の正解ラベルを取り出すことが
# できないかもしれない

# 学習済みモデルから確率密度関数と各成分の重み・偏差・単位ベクトルを取得
'''
prob_dist = gmm.predict_proba(data)
means = gmm.means_
covs = gmm.covariances_ # 共分散
weights_ = gmm.weights_
'''

# GMMからクラスタの予測を取得
predicted_labels = gmm.predict(train_data)

# 各クラスタの最も一般的な正解ラベルを保存する辞書
cluster_names = {}

# 各クラスタについてループ
for i in range(182):
    # このクラスタに分類された画像の正解ラベルの位置を取得(クラスタiに分類された画像のラベルを数字で取り出す)
    cluster_labels = train_labels_string_int[predicted_labels == i]
        
    # 最も一般的な正解ラベルの位置を取得
    most_common = np.bincount(cluster_labels).argmax()

    # 辞書に保存
    cluster_names[i] = ans_labels[most_common]

print(cluster_names)

predicted_labels = gmm.predict(test_data)

correct = 0
for i in range(182):

    # クラスタiに振り分けられた画像の正解ラベルの位置を取得
    cluster_labels = test_labels_string_int[predicted_labels == i]

    for j in range(len(cluster_labels)):
        if ans_labels[cluster_labels[j]] == cluster_names[i]:
            correct += 1
    
print((correct / (use_n - train_n)) * 100)



"""
26.419999999999998
33.52
30.98
37.68
38.14
40.96
41.28
https://aizine.ai/gaussian-mixture-model0627/
https://qiita.com/panda531/items/b0d209a253aa523561d9
https://qiita.com/maskot1977/items/1cdd80a29d8a0f995bc7#:~:text=%E3%82%AC%E3%82%A6%E3%82%B9%E6%B7%B7%E5%90%88%E3%83%A2%E3%83%87%E3%83%AB%EF%BC%88Gaussian%20Mixture,Model%3A%20GMM%EF%BC%89%E3%81%AF%E3%80%81%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8C%E8%A4%87%E6%95%B0%E3%81%AE%E3%82%AC%E3%82%A6%E3%82%B9%E5%88%86%E5%B8%83%EF%BC%88%E6%AD%A3%E8%A6%8F%E5%88%86%E5%B8%83%EF%BC%89%E3%81%AE%E9%87%8D%E3%81%AD%E5%90%88%E3%82%8F%E3%81%9B%E3%81%A7%E7%94%9F%E6%88%90%E3%81%95%E3%82%8C%E3%82%8B%E3%81%A8%E4%BB%AE%E5%AE%9A%E3%81%97%E3%81%9F%E7%A2%BA%E7%8E%87%E3%83%A2%E3%83%87%E3%83%AB%E3%81%A7%E3%81%99%E3%80%82
"""

