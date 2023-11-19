#ocrプログラム

#①ライブラリのインポート
import numpy as np    
import neurolab as nl

#②変数定義(use_n：読み込むデータ数 train_n：学習に使用するデータ数)
ans_labels="abcdefghijklmnopqrstuvwxyz"
use_n = 50000
train_n = int(0.9*use_n)
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
data = np.array(data).reshape(-1, pixel_n)

#⑤ニューラルネットを生成（域値:0~1,入力層(128),中間層(64),出力層(26)）
nn = nl.net.newff([[0, 1]] * pixel_n, [64, ans_n])
#学習方法としてRpropを指定
nn.trainf = nl.train.train_rprop
#⑥学習の実施
error_progress = nn.train(data[:train_n,:], labels[:train_n,:], 
                          epochs=50, show=10, goal=5000)

#学習素材でテスト
predicted = nn.sim(data[train_n:, :]) 
correct = 0
test_n = int(use_n-train_n)
for i in range(test_n): 
    if ans_labels[np.argmax(labels[train_n+i])]==ans_labels[np.argmax(predicted[i])]:
        correct+=1

print('正答率=',100*correct/test_n,'%')

#１文字画像の認識

from PIL import Image

# 手書き画像の読み込みとimageオブジェクトの作成
img = Image.open('a.png')
width, height = img.size
img2 = Image.new('RGB', (width, height))

 # getpixel((x,y))で左からx番目,上からy番目のピクセルの色を取得し、img_pixelsに追加する
img_pixels = []
for y in range(height):
    for x in range(width):
        img_pixels.append(img.getpixel((x,y)))

# データの規格化
img_pixels_norm = []
for i in range(8*16):
    p = img_pixels[i][0]
    if p == 0:
        img_pixels_norm.append(1.)
    else:
        img_pixels_norm.append(0.)

img_pixels_norm = np.array( img_pixels_norm).reshape(-1,8*16)

#結果表示
predicted = nn.sim( img_pixels_norm) 
print('結果:', ans_labels[np.argmax(predicted)]) 

"""
Epoch: 10; Error: 111627.71122374665;
Epoch: 20; Error: 34946.265763513176;
Epoch: 30; Error: 21416.338529024564;
Epoch: 40; Error: 18142.85489117686;
Epoch: 50; Error: 16768.702607686657;
The maximum number of train epochs is reached
正答率= 44.92 %
結果: a

Epoch: 10; Error: 126066.07089663418;
Epoch: 20; Error: 39249.20027310855;
Epoch: 30; Error: 20353.95356438003;
Epoch: 40; Error: 18220.62966105135;
Epoch: 50; Error: 17017.126335885187;
The maximum number of train epochs is reached
正答率= 42.02 %
結果: a

https://ebi-works.com/ocr-python/
"""