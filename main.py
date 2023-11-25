import pandas as pd
import numpy as np
import re
import os
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

from utils import preprocessing, create_dataset, get_plots
from create_dataset import vocab_generator, basic_training_set_generator, continual_training_set_generator
from model import model_architecture, model_training, prediction_and_performance
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

g1 = pd.read_excel(r'D:\ML-Projects\Continual-Learning\data\G1.xlsx')
g2 = pd.read_excel(r'D:\ML-Projects\Continual-Learning\data\G2.xlsx')
g3 = pd.read_excel(r'D:\ML-Projects\Continual-Learning\data\G3.xlsx')
g1.dropna(inplace=True)
g2.dropna(inplace=True)
g3.dropna(inplace=True)

g1['tags_cleaned'] = g1['tags'].apply(lambda x : preprocessing(x))
g2['tags_cleaned'] = g2['tags'].apply(lambda x : preprocessing(x))
g3['tags_cleaned'] = g3['tags'].apply(lambda x : preprocessing(x))

g1 = create_dataset(g1)
g2 = create_dataset(g2)
g3 = create_dataset(g3)

all_words, word2indx, indx2word, tag2indx, indx2tag = vocab_generator(g1, g2, g3)

X_tr_1, X_te_1, y_tr_1, y_te_1 = basic_training_set_generator(g1, word2indx, indx2word, tag2indx, indx2tag)
model_on_g1 = r"D:\ML-Projects\Continual-Learning\trained models\modeltrainedonG1.h5"
if os.path.exists(model_on_g1):
    # Code to execute if the file is present
    print(f"The file {model_on_g1} exists.")
    model = load_model(model_on_g1)
else:
    # Code to execute if the file is not present
    print(f"The file {model_on_g1} does not exist.")
    model = model_architecture(max_len= 250, all_words=all_words)
    model, history = model_training(model, X_tr_1, y_tr_1)
    get_plots(history)
    save_path = model_on_g1
    model.save(save_path)

prediction_and_performance(model, X_te_1, y_te_1, indx2tag)

X_tr_2, X_te_2, y_tr_2, y_te_2 = continual_training_set_generator(g1, g2, word2indx, indx2word, tag2indx, indx2tag)

model_on_g1_and_g2 = r"D:\ML-Projects\Continual-Learning\trained models\modeltrainedonG1andG2.h5"
if os.path.exists(model_on_g1_and_g2):
    # Code to execute if the file is present
    print(f"The file {model_on_g1_and_g2} exists.")
    model = load_model(model_on_g1_and_g2)
else:
    # Code to execute if the file is not present
    print(f"The file {model_on_g1_and_g2} does not exist.")
    model = model_architecture(max_len= 250, all_words=all_words)
    model, history = model_training(model, X_tr_2, y_tr_2)
    get_plots(history)
    save_path = model_on_g1_and_g2
    model.save(save_path)

prediction_and_performance(model, X_te_2, y_te_2, indx2tag)

X_tr_3, X_te_3, y_tr_3, y_te_3 = continual_training_set_generator(g2, g3, word2indx, indx2word, tag2indx, indx2tag)

model_on_g1_and_g2_and_g3 = r"D:\ML-Projects\Continual-Learning\trained models\modeltrainedonG1andG2andG3.h5"
if os.path.exists(model_on_g1_and_g2_and_g3):
    # Code to execute if the file is present
    print(f"The file {model_on_g1_and_g2_and_g3} exists.")
    model = load_model(model_on_g1_and_g2_and_g3)
else:
    # Code to execute if the file is not present
    print(f"The file {model_on_g1_and_g2_and_g3} does not exist.")
    model = model_architecture(max_len= 250, all_words=all_words)
    model, history = model_training(model, X_tr_3, y_tr_3)
    save_path = model_on_g1_and_g2_and_g3
    model.save(save_path)
    get_plots(history)
    print(f"Model saved to {save_path}")
prediction_and_performance(model, X_te_3, y_te_3, indx2tag)
