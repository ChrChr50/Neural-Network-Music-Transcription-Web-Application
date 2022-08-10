import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback, History
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape, Input
from keras.layers import LSTM
from keras.layers import BatchNormalization
import numpy as np
import glob
import pydub
import os
from .preprocessor2 import preprocess

max_length = 1000

raw_wav = []
raw_midi = []
train_wav = []
train_midi = []
test_wav = []
test_midi = []

# Convert all mp3 files to wav files
if glob.glob(r'C:\Users\chris\OneDrive\Documents\Cornell\Practice\AMT\maestro\2004\*.mp3') != []:
    for path1 in glob.glob(r'C:\Users\chris\OneDrive\Documents\Cornell\Practice\AMT\maestro\2004\*.mp3'):
        sound = pydub.AudioSegment.from_mp3(path1)
        sound.export(path1[:-3] + 'wav', format = 'wav')
        os.remove(path1)

# Enter wav and midi files into lists
for path2 in glob.glob(r'C:\Users\chris\OneDrive\Documents\Cornell\Practice\AMT\maestro\2004\*.midi'):
    raw_midi.append(path2)
    raw_wav.append(path2[:-4] + 'wav')

# Prepare input and output arrays
#batch = int(len(raw_wav) / 5)
batch = 5
temp1 = []
temp2 = []
for wavpath, midipath in zip(raw_wav[:batch], raw_midi[:batch]):
    input, output = preprocess(wavpath, midipath)
    temp1.append(np.array(input))
    temp2.append(np.array(output))

train_wav = temp1[:int(len(temp1) * 0.75)]
test_wav = temp1[int(len(temp1) * 0.75):]
train_midi = temp2[:int(len(temp2) * 0.75)]
test_midi = temp2[int(len(temp2) * 0.75):]

# Set up and train neural network
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, activation = 'relu', return_sequences = True, input_shape = (5, 84)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(512, activation = 'relu', return_sequences = True, input_shape = (5, 84)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(512, activation = 'relu'),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense((128 * max_length), activation = 'relu'),
    tf.keras.layers.Reshape((max_length, 128))
    ])

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
    )
EPOCHS = 100

trainer = model.fit(train_wav, train_midi, steps_per_epoch = 0, epochs = EPOCHS)
model.summary()
print(trainer.history['accuracy'])

# Test neural network
model.evaluate(test_wav, test_midi)

# Save trained model
model.save('music_model.h5')
