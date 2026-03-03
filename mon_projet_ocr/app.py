from flask import Flask, render_template, request, redirect
import os
from classifier import classify_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        doc_type, family, full_text = classify_image(filepath)

        return render_template(
            'result.html',
            type=doc_type,
            family=family,
            text=full_text[:500] + "..."
        )


if __name__ == '__main__':
    app.run(debug=True)