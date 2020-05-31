
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import numpy as np
from util import base64_to_pil
import test

# Declare a flask app
app = Flask(__name__)


def model_predict(img, model):
    print(img)
    preimage = img.resize((28,28))
    preimg = preimage.convert("L")
    img = np.reshape(preimg,(28,28))
    subimg = np.reshape(img,(-1,784))
    preds = model.predict(subimg)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        preds = model_predict(img, test)
        result = str(preds)
        return jsonify(result=result)
    print("end")
    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)

