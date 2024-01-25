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
with open('/content/drive/My Drive/OCR/letter.data', 'r') as f:
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

# 混合ガウス分布の作成（n_components=26, covariance_type='full'）
gmm = mixture.GaussianMixture(n_components=26, covariance_type='full')

# データセットの学習（fitメソッド）
gmm.fit(data)

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
predicted_labels = gmm.predict(data)

# 各クラスタの最も一般的な正解ラベルを保存する辞書
cluster_names = {}

# 正答率をためておく変数
ratio = 0

# 各クラスタについてループ
for i in range(26):
    # このクラスタに分類された画像の正解ラベルの位置を取得(クラスタiに分類された画像のラベルを数字で取り出す)
    cluster_labels = labels_string_int[predicted_labels == i]
    
    counts = np.bincount(cluster_labels)
    most_common = np.argmax(counts)
    most_common_count = counts[most_common]
    ratio += most_common_count / len(cluster_labels)

    # 最も一般的な正解ラベルの位置を取得
    most_common = np.bincount(cluster_labels).argmax()
    
    # 辞書に保存
    cluster_names[i] = ans_labels[most_common]

print(cluster_names)
print((ratio / 26) * 100)
"""
0: 't', 1: 'i', 2: 'p', 3: 'b', 4: 'e', 5: 'i', 6: 'n', 7: 'g', 8: 'o', 9: 'd', 10: 'e', 11: 'a', 12: 'n', 13: 'e', 14: 'o', 15: 'x', 16: 'h', 17: 'n', 18: 'u', 19: 'i', 20: 'o', 21: 'i', 22: 'p', 23: 'b', 24: 'a', 25: 'b'
0: 'n', 1: 'e', 2: 'i', 3: 'a', 4: 'g', 5: 'i', 6: 'i', 7: 'o', 8: 'g', 9: 'b', 10: 'o', 11: 'i', 12: 'd', 13: 'k', 14: 'u', 15: 'b', 16: 'e', 17: 'e', 18: 'b', 19: 'h', 20: 'n', 21: 'e', 22: 'g', 23: 'e', 24: 'n', 25: 'r'}

31.390937607228864(クラスタ1倍)
49.6009989970195(クラスタ4倍)
"""

"""
# 26個の混合ガウス分布はそれぞれどのアルファベットに対応しているのか
predicted_gaus =

# 確率密度関数から各サンプルがどの成分に属する確率を計算
# ここでは、各サンプルに対して最も高い確率を持つ成分のインデックスを取得します
max_indices = np.argmax(prob_dist, axis=1)

# 最大値が最も高い値に対応するサンプル番号（インデックス）から正解ラベル（出力データ）を取得
correct_labels = np.argmax(labels[max_indices], axis=1)

# GMMによって予想されたアルファベットを取得
predicted_alphabets = [ans_labels[i] for i in predicted_labels]

# 正解ラベルを取得
correct_alphabets = [ans_labels[np.argmax(label)] for label in labels]

# 正解ラベルと予想ラベルが一致する数を計算
correct_predictions = np.sum(np.array(predicted_alphabets) == np.array(correct_alphabets))

# 正答率を計算（一致する数 ÷ 全体の数）
accuracy = correct_predictions / len(correct_alphabets)

print(f"Accuracy: {accuracy * 100:.2f}%")



0.0
0.0
0.0
0.0
1.0461853620805432e-200
0.0
0.0
0.0
1.2623064742090215e-46
0.0
0.0
4.594059977437863e-172
0.0
0.0
9.146725743468767e-112
0.0
0.0
2.3028428853681007e-38
0.0
1.3657266587834822e-72
3.4332817047116815e-199
1.7607201148755332e-69
0.0
1.8730339249511167e-283
1.0
0.0
[0.12242074 0.18982091 0.30673958 0.40508881 0.44704201 0.41334191
 0.31086614 0.1705637  0.24002723 0.28886152 0.38720725 0.42090913
 0.45323146 0.44978837 0.41265502 0.25446252 0.31705965 0.33562668
 0.37345568 0.35832944 0.37758587 0.41952107 0.46216348 0.34111827
 0.38652489 0.34250354 0.31843243 0.27029491 0.29023342 0.36794994
 0.47661878 0.44015666 0.42778768 0.36175856 0.30605312 0.25997592
 0.27372907 0.34388012 0.46836981 0.53713831 0.42159882 0.31911827
 0.23452826 0.19669384 0.20082456 0.24209205 0.41540703 0.64924293
 0.42778643 0.26685191 0.17194105 0.15542867 0.1327335  0.13755077
 0.33080975 0.7812919  0.42228443 0.19738635 0.11072904 0.05708387
 0.03301233 0.04332981 0.22145887 0.89889878 0.44222918 0.16712495
 0.06121048 0.         0.         0.         0.1320486  1.
 0.44979347 0.1526833  0.04264098 0.         0.         0.
 0.08803027 0.96905262 0.46699067 0.13824344 0.03851438 0.
 0.         0.         0.16505904 0.91196973 0.46974204 0.14167784
 0.04470427 0.         0.         0.04470378 0.29985963 0.82668787
 0.48556421 0.20288826 0.07221442 0.         0.01925719 0.14717815
 0.46010129 0.68225746 0.47111265 0.3156898  0.17882277 0.08115615
 0.12723531 0.36657163 0.53508011 0.47800077 0.40783238 0.44841294
 0.40302271 0.39958848 0.46767573 0.53851485 0.46630078 0.28129062
 0.25928436 0.44360321 0.56533618 0.62242    0.58390562 0.43947666
 0.27028847 0.10522683]
[[0.1074349  0.04897646 0.01265497 ... 0.01772566 0.01505407 0.01325282]
 [0.04897646 0.15378993 0.09651972 ... 0.0321213  0.02434686 0.01854013]
 [0.01265497 0.09651972 0.21265141 ... 0.05157723 0.03263499 0.01036369]
 ...
 [0.01772566 0.0321213  0.05157723 ... 0.24633793 0.12949478 0.02596976]
 [0.01505407 0.02434686 0.03263499 ... 0.12949478 0.19723361 0.05065047]
 [0.01325282 0.01854013 0.01036369 ... 0.02596976 0.05065047 0.09415514]]
0.029080042827531915
"""

"""
https://aizine.ai/gaussian-mixture-model0627/
https://qiita.com/panda531/items/b0d209a253aa523561d9
https://qiita.com/maskot1977/items/1cdd80a29d8a0f995bc7#:~:text=%E3%82%AC%E3%82%A6%E3%82%B9%E6%B7%B7%E5%90%88%E3%83%A2%E3%83%87%E3%83%AB%EF%BC%88Gaussian%20Mixture,Model%3A%20GMM%EF%BC%89%E3%81%AF%E3%80%81%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8C%E8%A4%87%E6%95%B0%E3%81%AE%E3%82%AC%E3%82%A6%E3%82%B9%E5%88%86%E5%B8%83%EF%BC%88%E6%AD%A3%E8%A6%8F%E5%88%86%E5%B8%83%EF%BC%89%E3%81%AE%E9%87%8D%E3%81%AD%E5%90%88%E3%82%8F%E3%81%9B%E3%81%A7%E7%94%9F%E6%88%90%E3%81%95%E3%82%8C%E3%82%8B%E3%81%A8%E4%BB%AE%E5%AE%9A%E3%81%97%E3%81%9F%E7%A2%BA%E7%8E%87%E3%83%A2%E3%83%87%E3%83%AB%E3%81%A7%E3%81%99%E3%80%82
"""
