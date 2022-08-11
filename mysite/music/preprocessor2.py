import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import pretty_midi

# For initial training
def preprocess(wav, midi):

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
    unpacked_midi = pretty_midi.PrettyMIDI(midi)
    end_time = unpacked_midi.get_end_time()
    time_arr = np.linspace(0, end_time, input.shape[0])
    piano = unpacked_midi.get_piano_roll(fs = 84, times = time_arr).astype('float32')

    onehot = np.zeros((len(time_arr), 128))
    for note in unpacked_midi.instruments[0].notes:
        for time in range(int(note.start * (1 / (time_arr[1] - time_arr[0]))), int(note.end * (1 / (time_arr[1] - time_arr[0])))):
            onehot[time, note.pitch] = 1
    
    print('Preprocessing complete!')
    return input, onehot

# For production environment usage
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
