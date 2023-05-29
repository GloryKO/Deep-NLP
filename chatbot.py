import json
import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensoflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,GobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEconder   

#reading the json file

with open('intents.json') as file:
    data = json.load(file)
training_labels = []
taining_sentences =[]
labels =[]
responses =[]

for intent in data['intents']:
    for pattern in intent['patterns']:
        taining_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tags'] not in labels:
        labels.append(intent['tag'])

num_classes = len(labels)
