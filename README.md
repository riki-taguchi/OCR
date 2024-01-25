# OCR
Pythonを使ったOCRプログラムです。8×16ピクセルのアルファベット画像を読み取るOCRです。

ocr.py, ocr2.py, ocr2_1.py, ocr3.py のいずれかを実行すると、それぞれのOCRの正答率が出ます。

ocr.py はrpropを使った深層学習OCR、ocr2.py, ocr2_1.py, ocr3.py, ocr4_local.py は自作のアルゴリズムを使ったOCRとなっています。ocr2,3,4はそれぞれ平均法、k平均法、混合ガウス分布です。

フォルダ「OCRアプリ」は、フロントエンドでOCRを使えるようなアプリです。使用方法は次のようになります。

[使用方法]

1.リポジトリをクローンまたはダウンロードする

2.Pythonをインストールする

3.コマンドでpip install flask flask_cors pyodbcを実行する

4.ocr_flask.pyを実行してサーバーを起動する

5.ブラウザで http://localhost:5000 を開く

6.同じディレクトリの「画像サンプル」フォルダからファイルを選択する
