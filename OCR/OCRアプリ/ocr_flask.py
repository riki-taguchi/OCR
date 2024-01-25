from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import ocr2, ocr3, ocr4_local  # あなたのOCRスクリプト

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['test']
        if file:
            filename = secure_filename(file.filename)
            # サーバーに画像を保存
            file.save(os.path.join('.', filename))
            # 選択されたOCR関数を呼び出す
            ocr_choice = request.form.get('ocr')
            if ocr_choice == 'ocr2':
                result = ocr2.ocr2_program(filename)
            elif ocr_choice == 'ocr3':
                result = ocr3.ocr3_program(filename)
            elif ocr_choice == 'ocr4':
                result = ocr4_local.ocr4_program(filename)
            else:
                result = 'No OCR choice made'
            # 結果とフォームを同じページに表示
            return render_template('ocr.html', result=result)
        else:
            return 'No file part'
    return render_template('ocr.html')

if __name__ == '__main__':
    app.run(debug=True)
