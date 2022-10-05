import argparse
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import parallel_backend
import sys
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier

# Sys Arguments
#   [0] SRAT_TRAIN: training data are stratified
#   [1] Balanced weights: class_weight='balanced'
#   [2] Trained on geo: country data included in training set
#   [3] Hashed data: if yes, levels are hashed. if false, levels are ordinal
#   [4] BalancedTree: type of random forest classifier
#   [5] Model name: name of our modelrun

print("starting training")

strict = False
series_name = "SMOTE_Model"
if strict:
    series_name = series_name + "_Strict"


n_threads = 1

pwd = os.getcwd()
processed_path = os.path.join(pwd, os.path.relpath("../processed_data/", pwd))


# Select the data - Hash or int mapped
X = pd.read_csv(processed_path + series_name + "_int_mapping")
print("x data opened")

# If we're not training with geo data, drop these two colu
# We're ignoring keyErrors here; essentially, this is "drop if exists"
'''X = X.drop(columns=["plant_country_int", "pollinator_country_int",
                    "plant_country_hash", "pollinator_country_hash"], errors='ignore') '''

# turn DataFrame into a numpy matrix and normalize it
X = X.values
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
joblib.dump(scaler, series_name + "_Scaler")
print("x data transformed, scaler saved")

# read class values
y = pd.read_csv(processed_path + series_name+"_classes.csv").squeeze()
print("y data opened")
# Oversample for SMOTE Tree
oversample = SMOTE()
over_X, over_y = oversample.fit_resample(X, y)

# Split dataset into training set and test set (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(over_X, over_y, test_size=0.3, stratify=over_y)

# Create RFC
random_forest = RandomForestClassifier(n_estimators=200, verbose=1)
print("beginning training")
# Train the model using the training sets (Build a forest of trees from the training set (X, y))
with parallel_backend('threading', n_jobs=n_threads):
    random_forest.fit(X_train, y_train.values.ravel())

print("training complete")
# Save model

model_path = os.path.join(pwd, os.path.relpath("saved_models/"+series_name + "_model", pwd))

joblib.dump(random_forest, model_path)

print("model saved. evaluating model")
# Model prediction
y_pred = random_forest.predict(X_test)

# Evaluation Metrics
# Precision: True Positive / (True Positive + False Positive)
# Recall: True Positive / (True Positive + False Negative)
# F1: Harmonic mean between precision and recall

# Confusion Matrix: X axis = predicted, y = actual
#                                T              F
#                T        True Positive | False Negative
#                     __________________+__________________
#                F       False Positive | True Negative

# Evaluating our Model
recall = recall_score(y_test, y_pred)*100
precision = precision_score(y_test, y_pred)*100
f1 = f1_score(y_test, y_pred)*100
cm = confusion_matrix(y_test, y_pred)

# Write our results
write_path = os.path.join(pwd, os.path.relpath(("saved_models/model_evaluations/" + series_name + "_evaluation.txt"), pwd))

with open(write_path, "w+") as f:
    header = "\n* * * * * MODEL EVALUATION * * * * *"
    f.write(header)
    f.write("\n\t- "+str(n_threads)+" threads")

    f.write("\n\nRecall Score: {:.4f}".format(recall))
    f.write("\nPrecision Score: {:.4f}".format(precision))
    f.write("\nF1 Score: {:.4f}".format(f1))
    f.write("\nConfusion matrix: \n" + np.array2string(cm) + "\n\n")


