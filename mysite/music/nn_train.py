import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback
from keras import metrics, losses
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import LSTM, Conv2D, MaxPooling2D, add
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import glob
import pydub
import os
from .preprocessor2 import preprocess

subaudio = 0.1
batch = 20

raw_wav = []
raw_midi = []
train_wav = []
train_midi = []
test_wav = []
test_midi = []

# Convert all mp3 files to wav files
for path1 in glob.glob(r'..\maestro\2004\*.mp3'):
    sound = pydub.AudioSegment.from_mp3(path1)
    sound.export(path1[:-3] + 'wav', format = 'wav')
    os.remove(path1)

# Enter wav and midi files into lists
for path2 in glob.glob(r'..\maestro\2004\*.midi'):
    raw_midi.append(path2)
    raw_wav.append(path2[:-4] + 'wav')

# Prepare input and output arrays
temp1 = []
temp2 = []
for wavpath, midipath in zip(raw_wav[:batch], raw_midi[:batch]):
    input, output = preprocess(wavpath, midipath, subaudio)
    temp1.append(input)
    temp2.append(output)

train_wav = np.array(temp1[:int(len(temp1) * 0.75)])
test_wav = np.array(temp1[int(len(temp1) * 0.75):])
train_midi = np.array(temp2[:int(len(temp2) * 0.75)])
test_midi = np.array(temp2[int(len(temp2) * 0.75):])

train_wav.reshape(train_wav.shape[0], 1, int(22050 * subaudio))
test_wav.reshape(test_wav.shape[0], 1, int(22050 * subaudio))

# Set up and train neural network
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, activation = 'relu', return_sequences = True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(512, activation = 'relu', return_sequences = True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(12800 * subaudio),
    tf.keras.layers.Reshape(int(subaudio * 100), 128)
    ])

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
    )
EPOCHS = 100

trainer = model.fit(train_wav, train_midi, steps_per_epoch = 0, epochs = EPOCHS)

# Test neural network
model.evaluate(test_wav, test_midi)

# Save trained model
model.save('music_model.h5')
