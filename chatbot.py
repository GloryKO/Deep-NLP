import json
import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensoflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,GobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder   

#reading the json file

with open('intents.json') as file:
    data = json.load(file)
training_labels = []
training_sentences =[]
labels =[]
responses =[]

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tags'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)

# initializing the label encoder 
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
lbl_encoder = lbl_encoder.transform(training_labels)

# Tokenization
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences,truncating='post',maxlen=max_len)