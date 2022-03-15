import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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

# build model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, use_bias=False, kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(keras.layers.LeakyReLU(alpha=0.2)),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, use_bias=False, kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(keras.layers.LeakyReLU(alpha=0.2)),
    keras.layers.Dense(10, activation='softmax')
    ])

# compile model
optimizer = keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer, metrics=["accuracy"])

# callbacks (checkpoint, earlystopping, tensorboard)
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_log_dir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_log_dir()

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb, lr_scheduler])

best_model = keras.models.load_model("my_keras_model.h5")
best_model.evaluate(X_valid, y_valid)

# submission
test = pd.read_csv('data/test.csv')

# reshape data
test_img_data = []
for i in range(len(test)):
    test_img_data.append(test.iloc[i].values.reshape((28,28)))
test_img_data = np.array(test_img_data)

test_img_data = test_img_data/255.0

pred = model.predict(test_img_data)

pred_classes = np.argmax(pred, axis=1)
submission = pd.DataFrame(range(1,len(pred_classes)+1), columns=['ImageId'])
submission['Label'] = pred_classes
submission.to_csv('kaggle_submission.csv', index=False)
