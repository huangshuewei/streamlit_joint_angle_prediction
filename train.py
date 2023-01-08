# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 14:19:37 2023

@author: ASUS
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import keras
# from tensorflow.keras import regularizers

data = np.load(r"data\x_train_angles.npz")
x_train = data['a']
data.close()
data = np.load(r"data\x_test_angles.npz")
x_test = data['a']
data.close()
data = np.load(r"data\y_train_classes.npz")
y_train_class = data['a']
data.close()
data = np.load(r"data\y_test_classes.npz")
y_test_class = data['a']
data.close()

# one hot encode
onehot_encoder = OneHotEncoder(sparse=False)
y_train = y_train_class.reshape(-1, 1)
y_test = y_test_class.reshape(-1, 1)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

# build NN model
model = Sequential()
model.add(Dense(512, input_dim = 3,
                activation = 'relu',
                kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu',
                kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu',
                kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

optimizer = keras.optimizers.Adam(learning_rate=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

import tensorflow as tf
checkpointer = tf.keras.callbacks.ModelCheckpoint('NN_model_1.h5', verbose=1, save_best_only=True)


callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
            checkpointer]

history = model.fit(x_train, y_train, epochs=300,
                    batch_size=64,
                    validation_split=0.2,
                    validation_batch_size=64,
                    shuffle=True,
                    callbacks=callbacks)

# Evaluating model performance
loss, acc = model.evaluate(x_test, 
                           y_test, 
                           batch_size=32)


print('\n\nMODEL\n\nLoss     : {} \nAccuracy : {}'.format(history.history['loss'][-1],history.history['accuracy'][-1]))
print('\n\nMODEL\n\nLoss     : {} \nAccuracy : {}'.format(loss,acc))

'''
from sklearn.metrics import f1_score

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred,axis = 1) 
print(f1_score(y_test_class, y_pred_classes, average='macro'))
'''