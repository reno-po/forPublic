import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}
import numpy as np
import pandas as pd

from qrnn import QRNN
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout ,Activation
from keras import optimizers

from sklearn import datasets
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

length_of_sequences = 8
in_neurons          = 8
hidden_neurons      = 64
NUM_EPOCHS          = 30
dr                  = 0.1

digits = datasets.load_digits()
x = digits.data
y = digits.target

y = np_utils.to_categorical(y)

xr, xt, onehot_yr, onehot_yt = train_test_split(x, y, random_state=0)
xr = xr.reshape(-1,8,8)
xt = xt.reshape(-1,8,8)

model = Sequential()
model.add(QRNN(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_neurons), return_sequences=True))
model.add(Dense(hidden_neurons))
model.add(Dropout(dr))
model.add(QRNN(hidden_neurons, return_sequences=True))
model.add(Dense(hidden_neurons))
model.add(Dropout(dr))
model.add(QRNN(hidden_neurons, return_sequences=False))
model.add(Dense(10, activation='softmax')) #
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())
model.fit(xr, onehot_yr, batch_size=20, epochs=NUM_EPOCHS, validation_data=(xt, onehot_yt), verbose=1) #, callbacks=[early_stopping])
score = model.evaluate(xt, onehot_yt, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("\007")
