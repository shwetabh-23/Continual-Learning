import numpy as np
import pandas as pd
import random

from tensorflow.keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
from utils import clean_text

def vocab_generator(g1, g2, g3):
    all_words = []
    for i in range(len(g1['text'])):
        words = g1.iloc[i]['text'].split(' ')
        for word in words:
            all_words.append(clean_text(word))
            
    for i in range(len(g2['text'])):
        words = g2.iloc[i]['text'].split(' ')
        for word in words:
            all_words.append(clean_text(word))
            
    for i in range(len(g3['text'])):
        words = g3.iloc[i]['text'].split(' ')
        for word in words:
            all_words.append(clean_text(word))
            
    all_words = list(set(all_words))
    
    word2indx = {clean_text(word):i+2 for i, word in enumerate(all_words)}
    word2indx['UNK'] = 1
    word2indx['PAD'] = 0
    indx2word = {i:word for word, i in word2indx.items()}
    
    tag2indx = {'PAD' : 0,'treatment' : 1, 'chronic_disease' : 2, 'cancer' : 3, 'allergy' : 4, 'o' : 5}
    indx2tag = {i:word for word, i in tag2indx.items()}
    
    return all_words, word2indx, indx2word, tag2indx, indx2tag

def basic_training_set_generator(g1, word2indx, indx2word, tag2indx, indx2tag):
    max_len = 250
    X = [
        [word2indx.get(w[0], word2indx['UNK']) for w in s if w[0] in word2indx or 'UNK' in word2indx]
        for s in np.array(g1['all_mapping'])
    ]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word2indx["PAD"])
    
    all_tag = []
    for i, s in enumerate(np.array(g1['all_mapping'])):
        tag = []
        for w in (s):
            if w[1] == 'allergy_name':
                tag.append(tag2indx['allergy'])
            else:
                tag.append(tag2indx[w[1]])
        all_tag.append(tag)
    y = pad_sequences(maxlen=max_len, sequences=all_tag, padding="post", value=tag2indx["PAD"])
    y = [to_categorical(i, num_classes=6) for i in y]
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    #print(X_tr.shape, np.shape(y_tr))
    return X_tr, X_te, y_tr, y_te

def continual_training_set_generator(df1, df2, word2indx, indx2word, tag2indx, indx2tag):
    random_samples = random.sample(range(0, len(df1) + 1), 100)
    
    data_1 = df1.iloc[random_samples]
    data = pd.concat([data_1, df2], axis = 0)
    #print(data_1)
    X_tr, X_te, y_tr, y_te = basic_training_set_generator(data, word2indx, indx2word, tag2indx, indx2tag)
    return X_tr, X_te, y_tr, y_te
    