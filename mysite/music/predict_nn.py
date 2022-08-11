import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
import pydub
import pretty_midi
from .preprocessor2 import preprocess2

def wav_nn_predict(input):
    # Convert to wav file if needed
    if 'mp3' in input[-3:]:
        sound = pydub.AudioSegment.from_mp3(input)
        sound.export(input[:-3] + 'wav', format = 'wav')
        input = input[:-3] + 'wav'
    if 'ogg' in input[-3:]:
        sound = pydub.AudioSegment.from_ogg(input)
        sound.export(input[:-3] + 'wav', format = 'wav')
        input = input[:-3] + 'wav'

    # Use RNN to predict MIDI from wav file
    music_model = tf.keras.models.load_model('music_model.h5')
    processed_input = preprocess2(input)
    output = music_model.predict(processed_input)
    music_model.save()

    # Convert one hot encoded matrix into MIDI
    time = 0
    step = 0.3
    new_midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.instrument_name_to_program('Piano')
    piano_inst = pretty_midi.Instrument(program = piano)
    for note in output:
        target_note = pretty_midi.Note(velocity = 100, pitch = note, start = time, end = time + step)
        piano.notes.append(target_note)
        time += step
    new_midi.instruments.append(piano)

    return new_midi

# DELETE LATER
def wav_nn_predict2(input):
    print(input)
