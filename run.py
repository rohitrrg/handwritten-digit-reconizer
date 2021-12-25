from flask import Flask, app, render_template, request, redirect, flash, url_for
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')



if __name__=="__main__":
    app.run(port="8000")