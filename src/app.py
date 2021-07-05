# Importing Libraries & Packages
# Importing Libraries & Packages

import pandas as pd 
import numpy as np

import re, random, string
import itertools

import streamlit as st

# CountVectorizer will help calculate word counts
from sklearn.feature_extraction.text import CountVectorizer

# Import the string dictionary that we'll use to remove punctuation
import string

from rich.console import Console

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

import pickle
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras import initializers, regularizers, constraints, optimizers, layers

console = Console()

toxic_words_file = open("data/intermediate/toxic_words.pkl", "rb")
toxic_words_adj = pickle.load(toxic_words_file)

nontoxic_words_file = open("data/intermediate/nontoxic_words.pkl", "rb")
nontoxic_words_adj = pickle.load(nontoxic_words_file)

# loading tokenizer
with open('data/intermediate/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
model_1 = keras.models.load_model(filepath="models//toxic_lstm.h5")
maxpadlen = 200

tol = 0.001

def return_selected_text(tweet, prediction, tol = 0):
    tweet = tweet
    prediction = prediction
    
    if (prediction == 'nontoxic'):
        dict_to_use = nontoxic_words_adj # Calculate word weights using the pos_words dictionary
    elif (prediction == 'toxic'):
        dict_to_use = toxic_words_adj # Calculate word weights using the neg_words dictionary
        
    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
    
    score = 0
    selection_str = '' # This will be our choice
    lst = sorted(subsets, key = len) # Sort candidates by length
    #print(lst)
    
    
    for i in range(len(subsets)):
        
        new_sum = 0 # Sum for the current substring
        
        # Calculate the sum of weights for each word in the substring
        for p in range(len(lst[i])):
            if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
                new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]
            
        # If the sum is greater than the score, update our current selection
        if(new_sum > score + tol):
            score = new_sum
            selection_str = lst[i]
            #tol = tol*5 # Increase the tolerance a bit each time we choose a selection

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words
        
    return ' '.join(selection_str)


def toxicity_level(string):
    new_string = [string]
    new_string = tokenizer.texts_to_sequences(new_string)
    new_string = pad_sequences(new_string, maxlen=maxpadlen, padding='post')
    
    prediction = model_1.predict(new_string) #(Change to model_1 or model_2 depending on the preference of model type|| Model 1: LSTM, Model 2:LSTM-CNN)

    if prediction[0][0]> 0.50:
        pred = "toxic"
    else:
        pred = "nontoxic"


    return pred

def inference(text):
    #tokenize the text to get sentences
    s_list = nltk.tokenize.sent_tokenize(text)
    s_list = [x.replace('\n','').lower() for x in s_list]
    r_texts = []
    preds = []
    # make prediction on each sentence
    for sentence in s_list:
        #use model for prediction
        pred = toxicity_level(sentence)
        if pred == "toxic":
            preds.append(pred)
            r_texts.append(return_selected_text(sentence, pred, 0.001))
    dataf = pd.DataFrame(list(zip(s_list, preds, r_texts)), columns = ['Sentence', 'Prediction', 'Selected Text'])
    dataf.to_csv("outputs/file.csv", index = False)
    return r_texts

sample_text_2 = "such an asshole you are! just die whore. i don't wanna talk to you bitch."

st.image("NSFW Classifier--Text.png")
st.write(""" # Toxic Extraction""")
st.write("This is an explicit text detector which will identify explicit words in the text and will replace it with special characters!")

text = st.text_area('Enter Text Below (maximum 800 words):', height=300) 

submit = st.button('Generate') 

inference(sample_text_2)

if submit:

    st.write(""" ## Result:""")

    with st.spinner(text="This may take a moment..."):

        out = inference(text)
        tokens = set(list(itertools.chain(*[re.findall(f'\w+', x) for x in out])))
        fin_tokens = tokens - set(stopwords.words('english'))
        big_regex = re.compile('|'.join(map(re.escape, list(fin_tokens))))
        out_text = big_regex.sub('<removed>', text)
        
        
    console.print(fin_tokens)
    st.write(out_text)
