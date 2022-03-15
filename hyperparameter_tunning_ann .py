#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 09:47:24 2022

@author: rohit
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utiliti_functions import plot_history

# load mnist dataset
mnist = pd.read_csv("data/train.csv")

# saperate data and labels
data = mnist.drop(['label'], axis=1)
label = mnist['label']

# reshape data
img_data = []
for i in range(len(data)):
    img_data.append(data.iloc[i].values.reshape((28,28)))
img_data = np.array(img_data)

# split training and validation set
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in split.split(img_data, label):
    X_train = img_data[train_index]/255.0
    y_train = label[train_index]
    X_valid = img_data[val_index]/255.0
    y_valid = label[val_index]


from tensorflow import keras
import tensorflow as tf

# Hyperparameter Tunning
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[28,28]):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    model.add(keras.layers.Dense(10))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

keras_cl = keras.wrappers.scikit_learn.KerasClassifier(build_model)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [2,3,4,5],
    "n_neurons": np.arange(100, 500),
    "learning_rate": reciprocal(3e-4, 3e-2)}

rnd_search_cv = RandomizedSearchCV(keras_cl, param_distribs, n_iter=10, cv=5)
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
                  callbacks=[early_stopping_cb, checkpoint_cb])

rnd_search_cv.best_params_
rnd_search_cv.best_score_
rnd_search_cv.cv_results_