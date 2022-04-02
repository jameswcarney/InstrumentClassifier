import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import glob
import itertools
import librosa
import os
import pickle
import re
import scipy
import time

#directory to test data and json file
test_dir='audio_samples/nsynth-test/audio/'

"""
Get a list of the filenames for the test set from the json file
Drop anything that isn't bass, guitar, or strings (violin family)
Pickle the list for later use
"""
df_test_JSON = pd.read_json(path_or_buf='audio_samples/nsynth-test/examples.json', orient='index')

#Get all files, instead of a random sample
df_test_samples=df_test_JSON.groupby('instrument_family', as_index=False, group_keys=False).apply(lambda df: df)

# Drop everything except bass, guitar, and string (violin) families
df_test_samples = df_test_samples[df_test_samples['instrument_family']!=1] # brass
df_test_samples = df_test_samples[df_test_samples['instrument_family']!=2] # flute
df_test_samples = df_test_samples[df_test_samples['instrument_family']!=4] # keyboard
df_test_samples = df_test_samples[df_test_samples['instrument_family']!=5] # mallet
df_test_samples = df_test_samples[df_test_samples['instrument_family']!=6] # organ
df_test_samples = df_test_samples[df_test_samples['instrument_family']!=7] # reed
df_test_samples = df_test_samples[df_test_samples['instrument_family']!=9] # synth_lead
df_test_samples = df_test_samples[df_test_samples['instrument_family']!=10] # vocal

files_test = df_test_samples.index.tolist()

with open('Data/files_test.pickle', 'wb') as f:
    pickle.dump(files_test, f)


def extract_features(file):
    """
    Define function that takes in a file an returns features in an array
    Feature list:
    1. Harmonic vs Percussive
    2. Mel-Frequency Cepstral Coefficients
    3. Mel-Scaled Spectrogram
    4. Chroma Energy
    5. Spectral Contrast
    6. Zero-Crossing Rate
    7. Spectral Centroid
    8. Spectral Bandwidth
    9. Spectral Rolloff
    """

    y, sr = librosa.load(file)

    # Get the harmonic and percussive elements, then determine
    # whether the sample is primarily harmonic or primarily percussive
    # by comparing means
    y_harmonic = librosa.effects.harmonic(y)
    y_percussive = librosa.effects.percussive(y)

    if np.mean(y_percussive) > np.mean(y_harmonic):
        harmonic = 0
    else:
        harmonic = 1
    # Mel frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs, axis=1)

    # Mel spectrogram
    spectro = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectro = np.mean(spectro, axis=1)

    # Chroma energy (CENS)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma = np.mean(chroma, axis=1)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast, axis=1)

    # Zero-crossing rate
    # Should we actually take a sum of zero crossings for each sample? Not sure
    zero_crossings = librosa.zero_crossings(y)

    # spectral centroid
    centroids = librosa.feature.spectral_centroid(y, sr=sr, n_fft=1024, hop_length=512)
    centroids = np.mean(centroids, axis=1)

    # spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y, sr=sr)
    bandwidth = np.mean(bandwidth, axis=1)

    # spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y, sr=sr)
    rolloff = np.mean(rolloff, axis=1)

    return [harmonic, mfccs, spectro, chroma, contrast, zero_crossings, centroids, bandwidth, rolloff]


def get_instrument(filename):
    instrument_classes = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']

    for instrument_class in instrument_classes:
        if instrument_class in filename:
            return instrument_classes.index(name)
    else:
        return None

# This dict stores features for every sample
dict_test = {}
#loop over every file in the list
for file in files_test:
    #extract the features
    features = extract_features(test_dir + file + '.wav') #specify directory and .wav
    #add dictionary entry
    dict_test[file] = features

test_set_features = pd.DataFrame.from_dict(dict_test, orient='index', columns=['harmonic', 'mfcc', 'spectro', 'chroma',
                                                                           'contrast', 'zero_crossings', 'centroids',
                                                                           'bandwidth', 'rolloff'])

#extract mfccs
test_set_mfcc = pd.DataFrame(test_set_features.mfcc.values.tolist(), index=test_set_features.index)
test_set_mfcc = test_set_mfcc.add_prefix('mfcc_')

#extract spectro
test_set_spectro = pd.DataFrame(test_set_features.spectro.values.tolist(), index=test_set_features.index)
test_set_spectro = test_set_spectro.add_prefix('spectro_')


#extract chroma
test_set_chroma = pd.DataFrame(test_set_features.chroma.values.tolist(), index=test_set_features.index)
test_set_chroma = test_set_chroma.add_prefix('chroma_')


#extract contrast
test_set_contrast = pd.DataFrame(test_set_features.contrast.values.tolist(), index=test_set_features.index)
test_set_contrast = test_set_contrast.add_prefix('contrast_')

#zero crossings
test_set_zero_crossings = pd.DataFrame(test_set_features.zero_crossings.values.tolist(), index=test_set_features.index)
test_set_zero_crossings = test_set_zero_crossings.add_prefix('zero_crossings_')

#centroids
test_set_centroids = pd.DataFrame(test_set_features.centroids.values.tolist(), index=test_set_features.index)
test_set_centroids = test_set_centroids.add_prefix('centroids_')

#spectral bandwidth
test_set_bandwidth = pd.DataFrame(test_set_features.bandwidth.values.tolist(), index=test_set_features.index)
test_set_bandwidth = test_set_bandwidth.add_prefix('bandwidth_')

#spectral rolloff
test_set_rolloff = pd.DataFrame(test_set_features.rolloff.values.tolist(), index=test_set_features.index)
test_set_rolloff = test_set_rolloff.add_prefix('rolloff_')

#drop the old columns
test_set_features = test_set_features.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast', 'zero_crossings', 'centroids', 'bandwidth', 'rolloff'], axis=1)

#concatenate
df_test_features=pd.concat([test_set_features, test_set_mfcc, test_set_spectro, test_set_chroma, test_set_contrast, test_set_zero_crossings, test_set_centroids, test_set_bandwidth, test_set_rolloff],axis=1, join='inner')

# The target for each sample is the correct instrument family classification
# Derived from the filename
test_set_targets = []
for name in df_test_features.index.tolist():
    test_set_targets.append(get_instrument(name))

test_features['targets'] = test_set_targets

# Pickle it
with open('Data/df_test_features.pickle', 'wb') as f:
    pickle.dump(df_test_features, f)

# Extract the qualities provided by the NSynth dataset
test_qual = pd.DataFrame(df_test_samples.qualities.values.tolist(), index= df_test_samples.index)

json_test=pd.concat([df_test_samples, test_qual], axis=1, join='inner')

json_test= json_test.drop(labels=['qualities'], axis=1)

# Pickle it
with open('Data/json_test.pickle', 'wb') as f:
    pickle.dump(json_test, f)