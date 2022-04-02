import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import glob
import itertools
import librosa
import librosa.display
import os
import pickle
import re
import scipy
import time

"""
Wave File Exploration

We pick a single file from each of our three instrument families, then extract and graph the features of those samples
"""

bass = 'audio_samples/nsynth-valid/audio/bass_electronic_018-025-127.wav'
guitar = 'audio_samples/nsynth-valid/audio/guitar_acoustic_010-055-025.wav'
string = 'audio_samples/nsynth-valid/audio/string_acoustic_080-053-100.wav'

# Load each file
y_bass, sr_bass = librosa.load(bass, sr = None)
y_guitar, sr_guitar = librosa.load(guitar, sr = None)
y_string, sr_string = librosa.load(string, sr = None)

"""
Harmonic vs Percussive Elements
"""
fig, ax = plt.subplots(nrows=3, sharex=False, figsize=(10,12))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
# Bass
y_bass_harmonic = librosa.effects.harmonic(y_bass)
y_bass_percussive = librosa.effects.percussive(y_bass)

ax[0].set(title='Harmonic and Percussive Elements of Bass')
librosa.display.waveshow(y_bass_harmonic, color='tab:blue', sr=sr_bass, ax=ax[0], alpha=0.5, label='Harmonic')
librosa.display.waveshow(y_bass_percussive, color='tab:orange', sr=sr_bass, ax=ax[0], alpha=1, label='Percussive')
ax[0].legend()

# Guitar
y_guitar_harmonic = librosa.effects.harmonic(y_guitar)
y_guitar_percussive = librosa.effects.percussive(y_guitar)

ax[1].set(title='Harmonic and Percussive Elements of Guitar')
librosa.display.waveshow(y_guitar_harmonic, color='tab:blue', sr=sr_guitar, ax=ax[1], alpha=0.5, label='Harmonic')
librosa.display.waveshow(y_guitar_percussive, color='tab:orange', sr=sr_guitar, ax=ax[1], alpha=1, label='Percussive')
ax[1].legend()

# String
y_string_harmonic = librosa.effects.harmonic(y_string)
y_string_percussive = librosa.effects.percussive(y_string)

ax[2].set(title='Harmonic and Percussive Elements of Strings')
librosa.display.waveshow(y_string_harmonic, color='tab:blue', sr=sr_string, ax=ax[2], alpha=0.5, label='Harmonic')
librosa.display.waveshow(y_string_percussive, color='tab:orange', sr=sr_string, ax=ax[2], alpha=1, label='Percussive')
ax[2].legend()

plt.savefig('Plots/harmonic_percussive_exploration.png')

"""
Mel Frequency Cepstral Coefficients
A representation of timbre
"""
fig, ax = plt.subplots(nrows=3, sharex=False, figsize=(10,12))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

# Bass
ax[0].set(title='MFCCs For Bass')
mfccs_bass = librosa.feature.mfcc(y_bass, sr=sr_bass, n_mfcc=40)
first = librosa.display.specshow(mfccs_bass, x_axis='time', ax=ax[0])
plt.colorbar(first, ax=ax[0])

# Guitar
ax[1].set(title='MFCCs For Guitar')
mfccs_guitar = librosa.feature.mfcc(y_guitar, sr=sr_guitar, n_mfcc=40)
second = librosa.display.specshow(mfccs_guitar, x_axis='time', ax=ax[1])
plt.colorbar(second, ax=ax[1])

# Strings
ax[2].set(title='MFCCs For Strings')
mfccs_string = librosa.feature.mfcc(y_string, sr=sr_string, n_mfcc=40)
third = librosa.display.specshow(mfccs_bass, x_axis='time', ax=ax[2])
plt.colorbar(third, ax=ax[2])

plt.savefig('Plots/mfcc_exploration.png')

"""
Mel Spectrogram
A plot of the spectrum of frequences of a signal, on a mel scale
"""

fig, ax = plt.subplots(nrows=3, sharex=False, figsize=(10,12))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

#Bass
ax[0].set(title='Mel Spectrogram For Bass')
S = librosa.feature.melspectrogram(y=y_bass, sr=sr_bass, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
first = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_bass, fmax=8000, ax=ax[0])
fig.colorbar(first, ax=ax[0], format='%+2.0f dB')

#Guitar
ax[1].set(title='Mel Spectrogram For Guitar')
S = librosa.feature.melspectrogram(y=y_guitar, sr=sr_guitar, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
second = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_guitar, fmax=8000, ax=ax[1])
fig.colorbar(second, ax=ax[1], format='%+2.0f dB')

#Strings
ax[2].set(title='Mel Spectrogram For Strings')
S = librosa.feature.melspectrogram(y=y_string, sr=sr_string, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
third = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_guitar, fmax=8000, ax=ax[2])
fig.colorbar(third, ax=ax[2], format='%+2.0f dB')

plt.savefig('Plots/mel_spectrogram_exploration.png')

"""
Chroma Energy
Represents the strength of signal for each pitch class of the western 12-tone scale
"""
fig, ax = plt.subplots(nrows=3, sharex=False, figsize=(10,12))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

#Bass
ax[0].set(title='Chroma Energy For Bass')
chroma_bass = librosa.feature.chroma_cens(y=y_bass, sr=sr_bass)
first = librosa.display.specshow(chroma_bass, x_axis='time', y_axis='chroma', ax=ax[0])
plt.colorbar(first, ax=ax[0])

#Guitar
ax[1].set(title='Chroma Energy For Guitar')
chroma_guitar = librosa.feature.chroma_cens(y=y_guitar, sr=sr_guitar)
second = librosa.display.specshow(chroma_guitar, x_axis='time', y_axis='chroma', ax=ax[1])
plt.colorbar(second, ax=ax[1])
#Strings
ax[2].set(title='Chroma Energy For Strings')
chroma_string = librosa.feature.chroma_cens(y=y_string, sr=sr_string)
third = librosa.display.specshow(chroma_string, x_axis='time', y_axis='chroma', ax=ax[2])
plt.colorbar(third, ax=ax[2])

plt.savefig('Plots/chroma_energy_exploration.png')

"""
Spectral Contrast
"""
fig, ax = plt.subplots(nrows=3, sharex=False, figsize=(10,12))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

#Bass
ax[0].set(title='Spectral Contrast For Bass')
S=np.abs(librosa.stft(y_bass))
contrast_bass = librosa.feature.spectral_contrast(S=S, sr=sr_bass)
first = librosa.display.specshow(contrast_bass, x_axis='time', ax=ax[0])
plt.colorbar(first, ax=ax[0])
ax[0].set(ylabel='Frequency bands')

#Guitar
ax[1].set(title='Spectral Contrast For Guitar')
S=np.abs(librosa.stft(y_guitar))
contrast_guitar = librosa.feature.spectral_contrast(S=S, sr=sr_guitar)
second = librosa.display.specshow(contrast_guitar, x_axis='time', ax=ax[1])
plt.colorbar(second, ax=ax[1])
ax[1].set(ylabel='Frequency bands')

#Strings
ax[2].set(title='Spectral Contrast For Strings')
S=np.abs(librosa.stft(y_string))
contrast_string = librosa.feature.spectral_contrast(S=S, sr=sr_string)
third = librosa.display.specshow(contrast_string, x_axis='time', ax=ax[2])
plt.colorbar(third, ax=ax[2])
ax[2].set(ylabel='Frequency bands')

plt.savefig('Plots/spectral_contast_exploration.png')

"""
Zero-crossing Rate
"""

fig, ax = plt.subplots(nrows=3, sharex=False, sharey=False, figsize=(10,12))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

#Bass
zcrossing_bass = librosa.feature.zero_crossing_rate(y_bass)
ax[0].semilogy(zcrossing_bass.T, label='Rate')
ax[0].set(ylabel='Rate')
ax[0].set(xlabel='Frequency Bins')
ax[0].set(title="Zero-crossing Rate for Bass")
ax[0].legend()

#Guitar
zcrossing_guitar = librosa.feature.zero_crossing_rate(y_guitar)
ax[1].semilogy(zcrossing_guitar.T, label='Rate')
ax[1].set(ylabel='Rate')
ax[1].set(xlabel='Frequency Bins')
ax[1].set(title="Zero-crossing Rate for Guitar")
ax[1].legend()

#Strings
zcrossing_string = librosa.feature.zero_crossing_rate(y_string)
ax[2].semilogy(zcrossing_string.T, label='Rate')
ax[2].set(ylabel='Rate')
ax[2].set(xlabel='Frequency Bins')
ax[2].set(title="Zero-crossing Rate for Strings")
ax[2].legend()

plt.savefig('Plots/zero_crossing_rate_exploration.png')

"""
Spectral Centroid
Another timbral identifier, measuring sound brightness
"""
fig, ax = plt.subplots(nrows=3, sharex=False, sharey=False, figsize=(10,12))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
#Bass
centroid_bass = librosa.feature.spectral_centroid(y_bass)
ax[0].semilogy(centroid_bass.T, label='Frequency')
ax[0].set(ylabel='Frequency (Hz)')
ax[0].set(xlabel='Frequency Bins')
ax[0].set(title="Spectral Centroid for Bass")
ax[0].set(xlim=[0, centroid_bass.shape[-1]])
ax[0].legend()

#Guitar
centroid_guitar = librosa.feature.zero_crossing_rate(y_guitar)
ax[1].semilogy(zcrossing_guitar.T, label='Frequency')
ax[1].set(ylabel='Frequency (Hz)')
ax[1].set(xlabel='Frequency Bins')
ax[1].set(title="Zero-crossing Rate for Guitar")
ax[1].set(xlim=[0, centroid_guitar.shape[-1]])
ax[1].legend()

#Strings
centroid_string = librosa.feature.zero_crossing_rate(y_string)
ax[2].semilogy(zcrossing_string.T, label='Frequency')
ax[2].set(ylabel='Frequency (Hz)')
ax[2].set(xlabel='Frequency Bins')
ax[2].set(title="Zero-crossing Rate for Strings")
ax[2].set(xlim=[0, centroid_string.shape[-1]])
ax[2].legend()

plt.savefig('Plots/spectral_centroid_exploration.png')

"""
Spectral Bandwidth
Variance (in Hz) from the spectral centroid
"""
fig, ax = plt.subplots(nrows=3, sharex=False, sharey=False, figsize=(10,12))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

#Bass
bandwidth_bass = librosa.feature.spectral_bandwidth(y_bass, sr=sr_bass)
times = librosa.times_like(bandwidth_bass)
ax[0].semilogy(times, bandwidth_bass[0], label='Spectral Bandwidth')
ax[0].set(ylabel='Frequency (Hz)')
ax[0].set(xlabel='Time', xlim =[0, 4])
ax[0].legend()

#Guitar
bandwidth_guitar = librosa.feature.spectral_bandwidth(y_guitar, sr=sr_guitar)
times = librosa.times_like(bandwidth_guitar)
ax[1].semilogy(times, bandwidth_guitar[0], label='Spectral Bandwidth')
ax[1].set(ylabel='Frequency (Hz)')
ax[1].set(xlabel='Time', xlim =[0, 4])
ax[1].legend()

#Strings
bandwidth_string = librosa.feature.spectral_bandwidth(y_string, sr=sr_string)
times = librosa.times_like(bandwidth_string)
ax[2].semilogy(times, bandwidth_string[0], label='Spectral Bandwidth')
ax[2].set(ylabel='Frequency (Hz)')
ax[2].set(xlabel='Time', xlim =[0, 4])
ax[2].legend()

plt.savefig('Plots/spectral_bandwidth_exploration.png')

"""
Spectral Rolloff
Approximates the maximum frequency of a sample
This is mostly indicative of the pitch of the sample and overtones produced by the instrument
"""
fig, ax = plt.subplots(nrows=3, sharex=False, sharey=False, figsize=(10,12))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

#Bass
rolloff_bass = librosa.feature.spectral_rolloff(y_bass, sr=sr_bass, roll_percent = 0.95)[0]
ax[0].plot(rolloff_bass, label='Rolloff Frequency')
ax[0].set(ylabel='Frequency (Hz)')
ax[0].set(xlim=[0, rolloff_bass.shape[-1]])
ax[0].set(xlabel='Frequency Bins')
ax[0].set(title='Spectral Rolloff for Bass')
ax[0].legend()

#Guitar
rolloff_guitar = librosa.feature.spectral_rolloff(y_guitar, sr=sr_guitar, roll_percent = 0.95)[0]
ax[1].plot(rolloff_guitar, label='Rolloff Frequency')
ax[1].set(ylabel='Frequency (Hz)')
ax[1].set(xlim=[0, rolloff_guitar.shape[-1]])
ax[1].set(xlabel='Frequency Bins')
ax[1].set(title='Spectrall Rolloff for Guitar')
ax[1].legend()

#Strings
rolloff_string = librosa.feature.spectral_rolloff(y_string, sr=sr_string, roll_percent = 0.95)[0]
ax[2].plot(rolloff_string, label='Rolloff Frequency')
ax[2].set(ylabel='Frequency (Hz)')
ax[2].set(xlim=[0, rolloff_string.shape[-1]])
ax[2].set(xlabel='Frequency Bins')
ax[2].set(title='Spectrall Rolloff for Strings')
ax[2].legend()

plt.savefig('Plots/spectral_rolloff_exploration.png')


