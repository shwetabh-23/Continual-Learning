import pandas as pd
import numpy as np
import re

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

from utils import preprocessing, create_dataset
from create_dataset import vocab_generator, basic_training_set_generator, continual_training_set_generator


#Define Model : 
def model_architecture(max_len, all_words):
    # Input layer
    model_input = Input(shape=(max_len,))

    # Embedding layer
    embedded = Embedding(input_dim=len(all_words) + 2, output_dim=128, input_length=max_len, mask_zero=True)(model_input)

    # Bidirectional LSTM layer
    bi_lstm = Bidirectional(LSTM(units=256, return_sequences=True, recurrent_dropout=0.3))(embedded)

    # TimeDistributed Dense layer
    output = TimeDistributed(Dense(128, activation="softmax"))(bi_lstm)

    f_output = Dense(6, activation='relu')(output)
    # Create the model
    model = Model(model_input, f_output)

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Model summary
    print(model.summary())
    
    return model

def model_training(model, X_tr, y_tr):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    history = model.fit(X_tr, np.array(y_tr), batch_size=64, epochs=10,
                    validation_split=0.2, verbose=2)
    return model, history

def prediction_and_performance(model, X_te, y_te, indx2tag):
    y_pred = model.predict(X_te)
    y_pred = np.argmax(y_pred, axis = -1)
    y_test_true = np.argmax(y_te, axis = -1)
    
    y_pred = [[indx2tag[(i)] for i in row] for row in y_pred]
    y_test_true = [[indx2tag[(i)] for i in row] for row in y_test_true] 
    
    new_pred = []
    for preds in y_test_true:
        all_pred = []
        for pred in preds:
            if pred == "PAD":
                all_pred.append('o')
            else:
                all_pred.append(pred)
        new_pred.append(all_pred)
        
    #print('The F-1 score for this model is : ', f1_score(np.array(new_pred).ravel(), np.array(y_pred).ravel(), average = 'micro'))
    print(classification_report(np.array(new_pred).ravel(), np.array(y_pred).ravel()))
    return precision_recall_fscore_support(np.array(new_pred).ravel(), np.array(y_pred).ravel())