import os
from sys import base_exec_prefix
from flask import Flask, app, render_template, request, redirect, flash, url_for, jsonify
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt

init_Base64 = 21

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        draw = request.form['url']
        draw = draw[init_Base64:]
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        vect = np.asarray(resized, dtype="uint8")
        plt.imshow(vect, cmap='gray')
        plt.savefig('static/new_plot.png')

    return render_template('index.html', name='new_plot', url ='/static/new_plot.png')

if __name__=="__main__":
    app.run(port="8000", debug=True)