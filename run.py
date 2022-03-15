import os
from sys import base_exec_prefix
from flask import Flask, app, render_template, request, redirect, flash, url_for, jsonify
import cv2
import numpy as np
import base64
import matplotlib.pyplot as plt
import pickle
import keras

init_Base64 = 21

with open('model_pkl', 'rb') as f:
    ml_model = pickle.load(f)

ann_model = keras.models.load_model('my_keras_model.h5')

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
        image = np.asarray(bytearray(draw_decoded))
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
        print("resised shape:", resized.shape)
        vect = np.asarray(resized, dtype="uint8")
        plt.imshow(vect, cmap='gray')
        plt.savefig('static/plot.png')
        vect = vect.reshape(1,784)
        ml_result = ml_model.predict(vect)
        
        # Neural Networks
        img = (np.expand_dims(resized, 0))
        ann_result = ann_model.predict(img/255.0)
        
        

    return render_template('index.html', name='new_plot', url ='/static/plot.png',
                           ann_result = np.argmax(ann_result, axis=1)[0], 
                           ann_probs=ann_result.round(2), 
                           ml_result=ml_result[0])

if __name__=="__main__":
    app.run(port="8000", debug=True)