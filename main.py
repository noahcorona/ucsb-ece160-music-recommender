import warnings
import os
import math

from pprint import pprint
from prettytable import PrettyTable

import matplotlib.pyplot as plt

import numpy as np

from scipy.signal import stft
from scipy import interpolate

import librosa
import essentia
from essentia.standard import MonoLoader

"""
    Graph configuration
"""
plt.rcParams['figure.figsize'] = (20, 20)
plt.rcParams['figure.titlesize'] = 'large'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
axis_title_size = 20

"""
    Sample rate = 44.1kHz
"""

sample_rate, sample_freq = 44100, float(1/44100)


"""
    Feature extraction
"""

segment_size = 1024
hop_size = 512
window = np.hanning(segment_size)

media_dir = './minimusicspeech'
genres = [file for file in os.listdir(media_dir) if file is not 'speech']
audio_by_genre = dict()
file_count_by_genre = dict()

for idx, genre in enumerate(genres):
    audio_by_genre[genre] = dict()
    audio_files = [file for file in os.listdir(media_dir + genre)]
    num_files = len(audio_files)
    file_count_by_genre[genre] = num_files

    print('\n\nGENRE:', genre, '(' + str(num_files) + ' files)\n\n')

    for file in audio_files:
        # load audio file
        filepath = media_dir + genre + '/' + file
        loader = MonoLoader(filename=filepath)
        audio = loader()
        size = audio.size
        length_seconds = int(audio.size / sample_rate)
        audio_fft = np.fft.fft(audio)
        audio_by_genre[genre][file] = dict()
        audio_by_genre[genre][file]['path'] = filepath
        audio_by_genre[genre][file]['name'] = os.path.splitext(file)[0].capitalize()
        audio_by_genre[genre][file]['audio'] = audio[0:length_seconds*sample_rate]
        audio_by_genre[genre][file]['size'] = audio.size
        audio_by_genre[genre][file]['seconds'] = int(size / sample_rate)
        audio_by_genre[genre][file]['fft'] = audio_fft
        audio_by_genre[genre][file]['freq_counts'] = np.abs(audio_fft)

pprint(audio_by_genre)
pprint(file_count_by_genre)

