import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model

def wav_nn_predict(input):
    music_model = tf.keras.models.load_model('music_model.h5')
    output = music_model.predict(input)
    music_model.save()

# Convert path save to web server save