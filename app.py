import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import glob
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
from keras.models import load_model
from keras.preprocessing import image
import tensorflow_hub as hub
from flask import Flask,request,jsonify,render_template,url_for


new_model = tf.keras.models.load_model(
       ('cat_dog_model.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)


app = Flask(__name__)


def makeprediction(img_path):
    imagefile = image.load_img(img_path, target_size=(224,224))
    imagefile = image.img_to_array(imagefile)/255.0
    imagefile = imagefile.reshape(1,224,224,3)
    prediction = new_model.predict(imagefile)
    label = np.argmax(prediction)
    return label


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result',methods=['GET','POST'])
def result():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = 'static/'+img.filename
        img.save(img_path)

        pred = makeprediction(img_path)

    return render_template('result.html',output=pred)

if __name__=="__main__":
    app.run(debug=True)
