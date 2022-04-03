import pickle
import numpy as np
import pandas as pd
import librosa.display
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
import librosa
from sklearn.utils.multiclass import unique_labels

# file = "path/to/file.wav"

with open('Models/random_forest_RF.pickle', 'rb') as f:
    clf_rf = pickle.load(f)
    
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
    mfccs = np.mean(mfccs,axis=1)
    
    #Mel Spectrogram
    spectro = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)  
    spectro = np.mean(spectro, axis = 1)
    
    #compute chroma energy
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma = np.mean(chroma, axis = 1)
    
    #compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast, axis=1)

    #Zero-crossing rate
    zero_crossings = librosa.zero_crossings(y)

    #spectral centroid
    centroids = librosa.feature.spectral_centroid(y, sr=sr, n_fft=1024, hop_length=512)
    centroids = np.mean(centroids, axis=1)

    #spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y, sr=sr)
    bandwidth = np.mean(bandwidth, axis=1)

    #spectral rolloff
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


def classify_sample(file, features):
    dict_classify = {}
    dict_classify[file] = features
    
    features_classify = pd.DataFrame.from_dict(dict_classify, orient='index', columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast', 
                                                                                       'zero_crossings', 'centroids', 'bandwidth', 'rolloff'])
    
    #extract mfccs
    mfcc_classify = pd.DataFrame(features_classify.mfcc.values.tolist(), index=features_classify.index)
    mfcc_classify = mfcc_classify.add_prefix('mfcc_')

    #extract spectro
    spectro_classify = pd.DataFrame(features_classify.spectro.values.tolist(), index=features_classify.index)
    spectro_classify = spectro_classify.add_prefix('spectro_')


    #extract chroma
    chroma_classify = pd.DataFrame(features_classify.chroma.values.tolist(), index=features_classify.index)
    chroma_classify = chroma_classify.add_prefix('chroma_')


    #extract contrast
    contrast_classify = pd.DataFrame(features_classify.contrast.values.tolist(), index=features_classify.index)
    contrast_classify = contrast_classify.add_prefix('contrast_')

    #zero crossings
    zero_crossings_classify = pd.DataFrame(features_classify.zero_crossings.values.tolist(), index=features_classify.index)
    zero_crossings_classify = zero_crossings_classify.add_prefix('zero_crossings_')

    #centroids
    centroids_classify = pd.DataFrame(features_classify.centroids.values.tolist(), index=features_classify.index)
    centroids_classify = centroids_classify.add_prefix('centroids_')

    #spectral bandwidth
    bandwidth_classify = pd.DataFrame(features_classify.bandwidth.values.tolist(), index=features_classify.index)
    bandwidth_classify = bandwidth_classify.add_prefix('bandwidth_')

    #spectral rolloff
    rolloff_classify = pd.DataFrame(features_classify.rolloff.values.tolist(), index=features_classify.index)
    rolloff_classify = rolloff_classify.add_prefix('rolloff_')

    #drop the old columns
    features_classify = features_classify.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast', 'zero_crossings', 'centroids', 'bandwidth', 'rolloff'], axis=1)

    #concatenate
    df_features_classify=pd.concat([features_classify, mfcc_classify, spectro_classify, chroma_classify, contrast_classify, zero_crossings_classify, centroids_classify, bandwidth_classify, rolloff_classify],axis=1, join='inner')
    
    targets_classify = []
    targets_classify.append(get_instrument(file))
    
    df_features_classify['targets'] = targets_classify
    
    X_classify = df_features_classify.drop(labels=['targets'], axis=1)
    result = clf_rf.predict(X_classify)

    class_names = ['bass', 'brass', 'flute', 'guitar',
             'keyboard', 'mallet', 'organ', 'reed',
             'string', 'synth_lead', 'vocal']

    print(class_names[result[0]])

    return class_names[result[0]]


def graph_features(sample):
    y, sr = librosa.load(sample, duration=4)

    """
    Harmonic vs percussive
    """
    plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots()
    y_harm = librosa.effects.harmonic(y)
    y_perc = librosa.effects.percussive(y)

    plt.title("Harmonic and Percussive Components")
    librosa.display.waveshow(y_harm, sr=sr, color='tab:blue', ax=ax, alpha=0.5,  label='Harmonic')
    librosa.display.waveshow(y_perc, sr=sr, color='tab:orange', alpha=0.75, ax=ax , label='Percussive')
    ax.legend()
    plt.savefig('static/plots/harmonic_percussive.png')
    plt.close()

    """
    Chroma Energy
    """
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)

    plt.figure(figsize=(10, 4))
    plt.set_cmap('viridis')
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title("CENS")
    plt.savefig('static/plots/ChromaEnergy.png', bbox_inches='tight')
    plt.close()
    
    """
    Mel spectrogram
    """
    spectrogram_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    plt.figure(figsize=(10, 4))
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(spectrogram_mel, ref=np.max)
    img = librosa.display.specshow(S_dB, y_axis='mel', fmax=8000, x_axis='time', sr=sr, ax=ax)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel Spectrogram')
    plt.savefig('static/plots/mel_spectro.png', bbox_inches='tight')
    plt.close()
    
    """
    Spectral Contrast
    """
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(contrast, x_axis='time')
    plt.colorbar()
    plt.ylabel('Frequency bands')
    plt.title('Spectral contrast')
    plt.savefig('static/plots/spectral_contrast.png', bbox_inches='tight')
    plt.close()
    
    """
    MFCC
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    plt.figure(figsize=(10, 4))
    plt.set_cmap('viridis')
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    plt.savefig('static/plots/MFCC.png', bbox_inches='tight')
    plt.close()
    
    """
    Zero-crossing Rate
    """
    zcrossing = librosa.feature.zero_crossing_rate(y=y)

    plt.figure(figsize=(10, 4))
    plt.semilogy(zcrossing.T, label='Rate')
    plt.ylabel('Rate')
    plt.xlabel('Frequency Bins')
    plt.title('Zero-crossing Rate')
    plt.legend()
    plt.savefig('static/plots/zcrossing.png')
    plt.close()
    
    """
    Spectral Centroid
    """
    centroid = librosa.feature.spectral_centroid(y=y)

    plt.figure(figsize=(10, 4))
    plt.semilogy(centroid.T, label='Frequency')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Frequency Bins')
    plt.title("Spectral Centroid")
    plt.xlim([0, centroid.shape[-1]])
    plt.legend()    
    plt.savefig('static/plots/centroid.png')
    plt.close()
    
    """
    Spectral Bandwidth
    """
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    times = librosa.times_like(bandwidth)

    plt.figure(figsize=(10, 4))
    plt.semilogy(times, bandwidth[0], label='Spectral Bandwidth')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time')
    plt.title('Spectral Bandwidth')
    plt.xlim([0,4])
    plt.legend()
    plt.savefig('static/plots/spectral_bandwidth.png')
    plt.close()
    
    """
    Spectral Rolloff
    """
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent = 0.95)[0]

    plt.figure(figsize=(10, 4))
    plt.plot(rolloff, label="Rolloff Frequency")
    plt.ylabel('Frequency (Hz)')
    plt.xlim([0, rolloff.shape[-1]])
    plt.xlabel('Frequency Bins')
    plt.title('Spectral Rolloff')
    plt.legend()
    plt.savefig('static/plots/spectral_rolloff.png')
    plt.close()

features = extract_features(file)
graph_features(file)
classify_sample(file, features)