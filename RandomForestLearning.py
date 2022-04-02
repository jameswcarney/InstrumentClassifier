import numpy as np
import pandas as pd
import pickle
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import unique_labels

"""
Load training set
"""
with open('Data/df_training_features.pickle', 'rb') as f:
    df_training_set = pickle.load(f)

"""
Load test set
"""
with open('Data/df_test_features.pickle', 'rb') as f:
    df_test_set = pickle.load(f)

#get training and testing data
X_train = df_training_set.drop(labels=['targets'], axis=1)
X_test = df_test_set.drop(labels=['targets'], axis=1)

y_train = df_training_set['targets']
y_test = df_test_set['targets']

clf_randomforest =RandomForestClassifier(n_estimators=20, max_depth=50, warm_start=True)
clf_randomforest.fit(X_train, y_train)
y_pred_randomforest = clf_randomforest.predict(X_test)

accuracy_randomforest = np.mean(y_pred_randomforest == y_test)
print("Classifier accuracy is: {0:.2%}".format(accuracy_randomforest))

# pickle the trained model
with open("Models/random_forest_RF.pickle", mode='wb') as file:
    pickle.dump(clf_Rf, file)