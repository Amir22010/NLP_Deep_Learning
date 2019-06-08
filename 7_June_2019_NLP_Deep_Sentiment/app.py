# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:09:56 2019

@author: Amir.Khan
"""
from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 
import numpy as np
import time
app = Flask(__name__)
Bootstrap(app)

import torch
from string import punctuation
from SentimentRNN import SentimentRNN


# feel free to use this import 
from collections import Counter
## Build a dictionary that maps words to integers
# read data from text files
with open('reviews.txt', 'r') as f:
    reviews = f.read()

# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])
# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

# Load the trained model from file
net_save = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
trained_model = net_save
trained_model.load_state_dict(torch.load('model/sentiment.pth', map_location='cpu'))


def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])
    # splitting by spaces
    test_words = test_text.split()
    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])
    return test_ints


def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and 
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    
    return features


def predict(net, test_review, sequence_length=200):
    net.eval()
    # tokenize review
    test_ints = tokenize_review(test_review)
    # pad tokenized sequence
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)
    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)
    batch_size = feature_tensor.size(0)
    # initialize hidden state
    h = trained_model.init_hidden(batch_size)
    
    # get the output from the model
    output, h = trained_model(feature_tensor, h)
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))
#    # print custom response
#    if(pred.item()==1):
#        print("Positive review detected!")
#    else:
#        print("Negative review detected.")
        
    return output


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/analyse',methods=['POST'])
def analyse():
    start = time.time()
    if request.method == 'POST':
        seq_length=200
        prediction = []
        textsent = []
        score_value = []
        rawtext = request.form['rawtext']
        sent_tokenize_list = rawtext.split('.')
        while '' in sent_tokenize_list:
                sent_tokenize_list.remove('')
        for x in sent_tokenize_list:
            predictions = predict(net_save, x, seq_length)
            score_value.append(predictions.item())
            if predictions.item() > 0.5:
                results = "Positive"
                prediction.append(results)
            else:
                results = "Negative"
                prediction.append(results)
            textsent.append(x)
        end = time.time()
        final_time = end-start         
    return render_template('index.html',received_text = rawtext, result=prediction, textdisp =textsent, score=score_value,final_time=final_time)

if __name__ == '__main__':
	app.run(debug=True)