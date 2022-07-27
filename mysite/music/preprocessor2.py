import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import note_seq
import pretty_midi
import math

def preprocess(wav, midi, subaudio):

    # Generate spectrogram of wavfile
    amplitudes, sample_rate = librosa.load(wav)

    spec = np.abs(librosa.cqt(amplitudes, sample_rate))
    fig, ax = plt.subplots()
    diagram = librosa.display.specshow(librosa.amplitude_to_db(spec, ref = np.max),
        sr = sample_rate, x_axis = 'time', y_axis = 'cqt_hz', ax = ax)
    plt.colorbar(diagram, ax = ax, format = "%+2.0f dB")

    # Convert spectrogram into array for neural network
    window = 5
    segment = spec[:, :window]
    segment = np.pad(segment, ((0,0), (0,3)), 'constant')
    input = segment.T

    i = 1
    while (i + 1) * window < spec.shape[1]:
        segment = spec[:, (i * window):((i + 1) * window)]
        segment = np.pad(segment, ((0,0), (0,3)), 'constant')
        segment = segment.T
        input = np.concatenate((input, segment), axis = 0)
        i += 1

    segment = spec[:, (i * window):]
    segment = segment.T
    input = np.concatenate((input, segment), axis = 0)

    # Perform one hot encoding on MIDI
    seq = note_seq.midi_file_to_sequence_proto(midi)
    start = 0
    end = seq.notes[len(seq.notes) - 1].end_time
    time = np.arange(start, end, 0.01)
    onehot = np.zeros((len(time), 128))

    for note2 in seq.notes:
        pit = note2.pitch
        for j in range(int(note2.start_time * 100), int(note2.end_time * 100)):
            onehot[j, pit] = note2.velocity

    batches = math.ceil(len(amplitudes) / (sample_rate * subaudio))

    for z in range(int(batches * subaudio * 100) - len(onehot)):
        onehot = np.append(onehot, [np.zeros((128))], axis = 0)

    output = onehot.reshape(batches, int(subaudio * 100), 128)

    return input, output

def preprocess2(wav):

    # Generate spectrogram of wavfile
    amplitudes, sample_rate = librosa.load(wav)

    spec = np.abs(librosa.cqt(amplitudes, sample_rate))
    fig, ax = plt.subplots()
    diagram = librosa.display.specshow(librosa.amplitude_to_db(spec, ref = np.max),
        sr = sample_rate, x_axis = 'time', y_axis = 'cqt_hz', ax = ax)
    plt.colorbar(diagram, ax = ax, format = "%+2.0f dB")

    # Convert spectrogram into array for neural network
    window = 5
    segment = spec[:, :window]
    segment = np.pad(segment, ((0,0), (0,3)), 'constant')
    input = segment.T

    i = 1
    while (i + 1) * window < spec.shape[1]:
        segment = spec[:, (i * window):((i + 1) * window)]
        segment = np.pad(segment, ((0,0), (0,3)), 'constant')
        segment = segment.T
        input = np.concatenate((input, segment), axis = 0)
        i += 1

    segment = spec[:, (i * window):]
    segment = segment.T
    input = np.concatenate((input, segment), axis = 0)

    return input

# DEPRECATED
def preprocess_alt(wav):

    # Generate spectrogram of wavfile
    amplitudes, sample_rate = librosa.load(wav)

    spec = np.abs(librosa.cqt(amplitudes, sample_rate))
    fig, ax = plt.subplots()
    diagram = librosa.display.specshow(librosa.amplitude_to_db(spec, ref = np.max),
        sr = sample_rate, x_axis = 'time', y_axis = 'cqt_hz', ax = ax)
    plt.colorbar(diagram, ax = ax, format = "%+2.0f dB")

    # Convert spectrogram into array for neural network
    newspec = spec.T
    for vector in newspec:
        magnitude = np.linalg.norm(vector)
        if magnitude > 0:
            vector = vector / magnitude

    return newspec