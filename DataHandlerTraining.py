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

#directory to training data and json file
training_dir='audio_samples/nsynth-train/audio/'

"""
Get a list of the filenames for the training set from the json file
Drop anything that isn't bass, guitar, or strings (violin family)
Pickle the list for later use
"""

#read the raw json files as given in the training set
df_train_JSON = pd.read_json(path_or_buf='audio_samples/nsynth-train/examples.json', orient='index')

#Sample 5000 files
df_train_samples=df_train_JSON.groupby('instrument_family', as_index=False, group_keys=False).apply(lambda df: df.sample(5000))

# Drop everything except bass, guitar, and string (violin) families
df_train_samples = df_train_samples[df_train_samples['instrument_family']!=1] # brass
df_train_samples = df_train_samples[df_train_samples['instrument_family']!=2] # flute
df_train_samples = df_train_samples[df_train_samples['instrument_family']!=4] # keyboard
df_train_samples = df_train_samples[df_train_samples['instrument_family']!=5] # mallet
df_train_samples = df_train_samples[df_train_samples['instrument_family']!=6] # organ
df_train_samples = df_train_samples[df_train_samples['instrument_family']!=7] # reed
df_train_samples = df_train_samples[df_train_samples['instrument_family']!=9] # synth_lead
df_train_samples = df_train_samples[df_train_samples['instrument_family']!=10] # vocal

#save the train file index as list
files_training = df_train_samples.index.tolist()

#save the list to a pickle file
with open('Data/files_training.pickle', 'wb') as f:
    pickle.dump(files_training, f)

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

    # Load the file for feature extraction
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

    # Mel-Frequency Cepstral Coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs, axis=1)

    # Mel Spectrogram
    spectro = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectro = np.mean(spectro, axis=1)

    # Chroma Energy (CENS)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma = np.mean(chroma, axis=1)

    # Spectral Contrast
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
dict_training = {}
#loop over every file in the list
for file in files_training:
    #extract the features
    features = extract_features(training_dir + file + '.wav') #specify directory and .wav
    #add dictionary entry
    dict_training[file] = features

training_set_features = pd.DataFrame.from_dict(dict_training, orient='index', columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast', 'zero_crossings', 'centroids', 'bandwidth', 'rolloff'])

#extract mfccs
training_set_mfcc = pd.DataFrame(training_set_features.mfcc.values.tolist(), index=training_set_features.index)
training_set_mfcc = training_set_mfcc.add_prefix('mfcc_')

#extract spectro
training_set_spectro = pd.DataFrame(training_set_features.spectro.values.tolist(), index=training_set_features.index)
training_set_spectro = training_set_spectro.add_prefix('spectro_')


#extract chroma
training_set_chroma = pd.DataFrame(training_set_features.chroma.values.tolist(), index=training_set_features.index)
training_set_chroma = training_set_chroma.add_prefix('chroma_')


#extract contrast
training_set_contrast = pd.DataFrame(training_set_features.contrast.values.tolist(), index=training_set_features.index)
training_set_contrast = training_set_contrast.add_prefix('contrast_')

#zero crossings
training_set_zero_crossings = pd.DataFrame(training_set_features.zero_crossings.values.tolist(), index=training_set_features.index)
training_set_zero_crossings = training_set_zero_crossings.add_prefix('zero_crossings_')

#centroids
training_set_centroids = pd.DataFrame(training_set_features.centroids.values.tolist(), index=training_set_features.index)
training_set_centroids = training_set_centroids.add_prefix('centroids_')

#spectral bandwidth
training_set_bandwidth = pd.DataFrame(training_set_features.bandwidth.values.tolist(), index=training_set_features.index)
training_set_bandwidth = training_set_bandwidth.add_prefix('bandwidth_')

#spectral rolloff
training_set_rolloff = pd.DataFrame(training_set_features.rolloff.values.tolist(), index=training_set_features.index)
training_set_rolloff = training_set_rolloff.add_prefix('rolloff_')

training_set_features = training_set_features.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast', 'zero_crossings', 'centroids', 'bandwidth', 'rolloff'], axis=1)
df_training_features=pd.concat([training_set_features, training_set_mfcc, training_set_spectro, training_set_chroma, training_set_contrast, training_set_zero_crossings, training_set_centroids, training_set_bandwidth, training_set_rolloff],axis=1, join='inner')

# The target for each sample is the correct instrument family classification
# Derived from the filename
training_set_targets = []
for name in df_features_training.index.tolist():
    training_set_targets.append(get_instrument(name))

training_features['targets'] = training_set_targets

# Pickle it
with open('Data/df_training_features.pickle', 'wb') as f:
    pickle.dump(df_training_features, f)

#%%
# Extract the qualities provided by the NSynth dataset
training_qual = pd.DataFrame(df_train_samples.qualities.values.tolist(), index= df_train_samples.index)

json_training=pd.concat([df_train_samples, training_qual],
                           axis=1, join='inner')

json_training= json_training.drop(labels=['qualities'], axis=1)

# Pickle it
with open('Data/json_training.pickle', 'wb') as f:
    pickle.dump(json_training, f)