# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:11:18 2019

@author: Amir
"""
from flask import Flask , render_template ,request
from pickle import load 
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences 
from keras.applications.vgg16 import VGG16 
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.models import Model 
from keras.models import load_model
from keras import backend as K

app = Flask(__name__)


tokenizer = load(open('models/tokenizer.pkl', 'rb'))
max_length = 34 

def extract_features(filename):
	K.clear_session()
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	K.clear_session()
	return feature


def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None


def cleanup_summary(summary):
  index = summary.find('startseq ')
  if index > -1:
    summary = summary[len('startseq '):]
  index = summary.find(' endseq')
  if index > -1:
    summary = summary[:index]
  return summary

def generate_desc(model, tokenizer, photo, max_length):
  in_text = 'startseq'
  for _ in range(max_length):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    yhat = model.predict([photo,sequence], verbose=0)
    yhat = argmax(yhat)
    word = word_for_id(yhat, tokenizer)
    if word is None:
      break
    in_text += ' ' + word
    if word == 'endseq':
      break
  return in_text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/success", methods=['POST'])
def upload_file():
    target_image_path = request.form.get('target')
    photo = extract_features(target_image_path)
    global model
    model = load_model('models/model_weight.h5')
    description = generate_desc(model, tokenizer, photo, max_length)
    description = cleanup_summary(description)
    return render_template('success.html', image_to_show = '../'+target_image_path, result = description, init=True)
    
if __name__ =="__main__":
	app.run(debug=True)

















