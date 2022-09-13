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
#   [5] Model name: name of our model

n_threads = 3

parser = argparse.ArgumentParser()
parser.add_argument('-CT', '--CONTROL', required=False, type=bool, default=False)
parser.add_argument('-WB', '--WEIGHT_BALANCE', required=False, type=bool, default=False)
parser.add_argument('-GT', '--GEO_TRAIN', required=False, type=bool, default=False)
parser.add_argument('-HD', '--HASHED_DATA', required=False, type=bool, default=False)
parser.add_argument('-SM', '--SMOTE_TREE', required=False, type=bool, default=False)
parser.add_argument('-n', '--MODEL_NAME', required=True)

args = parser.parse_args()

pwd = os.getcwd()
processed_path = os.path.join(pwd, os.path.relpath("../processed_data", pwd))


# Select the data - Hash or int mapped
if args.HASHED_DATA:
    X = pd.read_csv(processed_path + "/Pollinator_Plant_hash_mapped.csv")
else:
    X = pd.read_csv(processed_path + "/Pollinator_Plant_int_mapped.csv")

# If we're not training with geo data, drop these two columns
if not args.GEO_TRAIN:
    # We're ignoring keyErrors here; essentially, this is "drop if exists"
    X = X.drop(columns=["plant_country_int", "pollinator_country_int",
                        "plant_country_hash", "pollinator_country_hash"], errors='ignore')


X = X.values  # turn DataFrame into a numpy matrix
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

y = pd.read_csv(processed_path + "/classes.csv").squeeze()

# Split dataset into training set and test set (70% train, 30% test)
if args.SMOTE_TREE:
    # If we're doing a smote tree, oversample
    oversample = SMOTE()
    over_X, over_y = oversample.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(over_X, over_y, test_size=0.1, stratify=over_y)

elif not args.CONTROL: # Only one case we will not stratify data #TODO might want to remove this check
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


if args.WEIGHT_BALANCE:
    random_forest = RandomForestClassifier(n_estimators=200, class_weight='balanced', verbose=1)
else:
    random_forest = RandomForestClassifier(n_estimators=200, verbose=1)

# Train the model using the training sets (Build a forest of trees from the training set (X, y))
with parallel_backend('threading', n_jobs=n_threads):
    random_forest.fit(X_train, y_train.values.ravel())

# Save model
model_path = os.path.join(pwd, os.path.relpath(("saved_models/"+args.MODEL_NAME), pwd))
joblib.dump(random_forest, model_path)

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
write_path = os.path.join(pwd, os.path.relpath(("saved_models/model_evaluations/"+args.MODEL_NAME+".txt"), pwd))

with open(write_path, "w+") as f:
    header = "\n* * * * * " + args.MODEL_NAME + " Evaluation * * * * *"
    f.write(header)
    f.write("\n\t- "+str(n_threads)+" threads")
    f.write("\n\t- Stratified: "+str(not args.CONTROL))
    f.write("\n\t- Weight Balanced: "+str(args.WEIGHT_BALANCE))
    f.write("\n\t- Trained on countries: "+str(args.GEO_TRAIN))
    f.write("\n\t- Data hashed: "+str(args.HASHED_DATA))
    f.write("\n\t- Smote Tree: "+str(args.SMOTE_TREE))

    f.write("\n\nRecall Score: {:.4f}".format(recall))
    f.write("\nPrecision Score: {:.4f}".format(precision))
    f.write("\nF1 Score: {:.4f}".format(f1))
    f.write("\nConfusion matrix: \n" + np.array2string(cm) + "\n\n")

print(args.MODEL_NAME, "finished running")

