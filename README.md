# InstrumentClassifier
A random forest classifier for identifying musical instruments in audio recordings. This project was created 
to fulfill senior design project requirements for my Bachelor of Computer Science degree.

A working demonstration of this classifier is available at http://jamescarney.pythonanywhere.com




## Features

- Classification of bass, guitar, or string (violin) family instruments
- Can be modified to include all instrument families in the NSynth dataset
- Utilizes a random forest classifier



## Installation

Create a new Python virtualenv in the desired directory. 

```bash
  python3 -m venv InstrumentClassifier
```
Copy the contents of this repository to \InstrumentClassifier

CD to the project directory

Activate the virtual environment. On Windows, run:

```bash
  InstrumentClassifier\Scripts\activate.bat
```

Install dependencies from the requirements.txt included with the source:

```bash
  pip install -r requirements.txt
```

## Usage

If you wish to run the scripts to generate a classifier, you will need to place the NSynth dataset in the audio_samples folder.
Files are available at: https://magenta.tensorflow.org/datasets/nsynth#files. 

Copy the nsynth_test, nsynth_training, and nsynth_valid
folders into audio_samples.

To create a classifier, you must first run DataHandlerTest.py and DataHandlerTraining.py to extract features from the audio for 
model training. Once this is done, you can train the model by running RandomForestLearning.py. However, a trained classifier
is already provided in /Models so you may avoid the 7-8 hour process of extracting features and training the model.

The completed model can be run on an audio sample by running classifier.py. To choose the file, edit line 12 to include the
file path and remove the '#'. I recommend the Nsynth-test dataset, as the files are cleaned and normalized in the same manner as
the set used to train the model, and will therefore deliver the best results. Classifier.py prints the classification result to 
console, and also provides plots of the extracted features in /Plots.


## Authors

- [@jameswcarney](https://www.github.com/jameswcarney)
