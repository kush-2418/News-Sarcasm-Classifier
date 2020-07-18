#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:04:34 2020

@author: kush
"""

from flask import Flask,render_template,request
import pickle
import os
from numpy import array
import nltk
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
import string
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


tokenizer = pickle.load(open('tokenizer.pkl','rb'))
sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
model = load_model('SarcasmLSTM_model.h5')
stop_words = set(nltk.corpus.stopwords.words('english'))
punctuation = list(string.punctuation)
stop_words.update(punctuation)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

def clean_text(text):

    lemmatizer = WordNetLemmatizer()
    soup = BeautifulSoup(text,'html.parser').get_text()
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", '', text)
    text = re.sub('[^A-Z a-z 0-9-]+',' ',text)
    text = re.sub('[.!-]+',' ',text)
    text = re.sub(r'http\S+','',text)
    text = re.sub('[\d]','',text)
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words])
    
    return text

def predict_sarcasm(sample_review):
    maxlen = 60
    sample_review = clean_text(sample_review)
    sample_review = [sample_review]
    seq = tokenizer.texts_to_sequences(sample_review)
    padded = pad_sequences(seq, maxlen=maxlen)
    with graph.as_default():
        set_session(sess)
        pred = model.predict_classes(padded)
    return pred

@app.route('/get_sarcasm',methods=['POST','GET'])
def get_sarcasm():
    if request.method =='POST':
        text = request.form['text']
        prediction = predict_sarcasm(text)
        if prediction == 1:
            prediction = 'The News is Sarcastic'
        else:
            prediction = 'The News has No Sarcasm'
            
    return render_template('index.html', sarcasm = prediction)


if __name__ == '__main__':
    app.run()
