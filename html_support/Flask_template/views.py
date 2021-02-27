from flask import render_template, request
from flask import redirect, url_for
import pickle
import os
from PIL import Image
from utils import pipeline_model

UPLOAD_FOLDER = 'static/uploads'


def base():
    return render_template("base.html")


def index():
    return render_template("index.html", upload=False)


def getwidth(path):
    img = Image.open(path)
    size = img.size  # width and height
    aspect = size[0] / size[1]  # width / height
    w = 300 * aspect
    return int(w)


def faceapp():
    if request.method == 'POST':
        file = request.files['image']
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        print('File saved successfully...')
        # Processing
        w = getwidth(path)

        # Predict now: Pass image to pipeline model
        pipeline_model(path, file.filename, color='bgr')

        return render_template('faceapp.html', fileUpload=True, img_name=file.filename, w=w)
    return render_template('faceapp.html', fileUpload=False, img_name="xyz.png", w="280")
