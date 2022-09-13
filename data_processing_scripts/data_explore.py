import hashlib

import argparse as argparse
import numpy as np
import pandas as pd
import requests
import re
import category_encoders as ce
import os
import sys
import argparse

import requests as requests
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher, DictVectorizer


def hash_str(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest(), "big")


def getNums(substr):
    return re.findall(r'\d+', substr)[0]


def n_nonzero_columns(X):
    return len(np.unique(X.nonzero()[1]))


def package(f):
    arr = [f]
    return arr
'''
STRAT_TRAIN = sys.argv[0]
WEIGHT_BALANCE = sys.argv[1]
GEO_TRAIN = sys.argv[2]
HASHED_DATA = sys.argv[3]
BALANCED_TREE = sys.argv[4]
MODEL_NAME = sys.argv[5]
'''

pwd = os.path.dirname(__file__)

class_path = os.path.join(pwd, os.path.relpath("processed_data/classes.csv", pwd))
int_mapped_path = os.path.join(pwd, os.path.relpath("processed_data/Pollinator_Plant_int_mapped.csv", pwd))
poll_plant_path = os.path.join(pwd, os.path.relpath("processed_data/Pollinator_Plant_hash_mapped.csv", pwd))
corrupted_path = os.path.join(pwd, os.path.relpath("processed_data/Corrupted_PollPlant.csv", pwd))


classes = pd.read_csv(class_path)
int_mapped = pd.read_csv(int_mapped_path)
hash = pd.read_csv(poll_plant_path)
corrupted = pd.read_csv(corrupted_path)

print("int mapped: ",int_mapped.columns)
print("hash mapped: ", hash.columns)
print("corrupted: ",corrupted.columns)

print(int_mapped.columns == hash.columns)
print(hash.columns == corrupted.columns)
print(int_mapped.columns == corrupted.columns)


#print(classes['class'].value_counts())
#print(classes.shape)

#print(int_mapped.shape)
#print(hash.shape)

#print(int_mapped.columns)
#print(hash.columns)


breakpoint()


# If an argument is set to any string, it will be marked as true. Otherwise it will be false
parser = argparse.ArgumentParser()
parser.add_argument('-ST', '--STRAT_TRAIN', required=False, type=bool, default=False)
parser.add_argument('-WB', '--WEIGHT_BALANCE', required=False, type=bool, default=False)
parser.add_argument('-GT', '--GEO_TRAIN', required=False, type=bool, default=False)
parser.add_argument('-HD', '--HASHED_DATA', required=False, type=bool, default=False)
parser.add_argument('-BT', '--BALANCED_TREE', required=False, type=bool, default=False)
parser.add_argument('-MN', '--MODEL_NAME', required=False, type=bool, default=False)

args = parser.parse_args()



breakpoint()

r = requests.get("https://api.gbif.org/v2/map/occurrence/density/0/0/0@1x.png?taxonKey=212")
print(r.status_code)

print(r.json)
print(r.text)

breakpoint()

pwd = os.path.dirname(__file__)
source_data_path = os.path.join(pwd, os.path.relpath("source_data/PollPlantsGBIF2.csv", pwd))

int_mapped_path = os.path.join(pwd, os.path.relpath("processed_data/Pollinator_Plant_int_mapped.csv", pwd))
poll_plant_path = os.path.join(pwd, os.path.relpath("processed_data/Pollinator_Plant_Interactions.csv", pwd))

int_mapped_df = pd.read_csv(int_mapped_path, nrows=100)
print(int_mapped_df.columns)

poll_plant_df = pd.read_csv(poll_plant_path, nrows=100)
print(poll_plant_df.columns)


breakpoint()
classes = pd.read_csv("../processed_data/classes.csv")
classes.reset_index(inplace=True, drop=True)
print(classes.loc[classes["class"]==1].count())
print(classes.loc[classes["class"]==0].count())

breakpoint()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

PollPlants = pd.read_csv("../source_data/PollPlantsGBIF2.csv")
PollPlants.drop(columns=["pollinator_scientificName", "class", "plant_scientificName"], inplace=True)

ixn = pd.read_csv("../processed_data/Pollinator_Plant_Interactions.csv")

ixn["plant_taxonKey"] = ixn["plant_taxonKey"].apply(lambda str: getNums(str))
ixn["pollinator_taxonKey"] = ixn["pollinator_taxonKey"].apply(lambda str: getNums(str))

ixn.to_csv("Pollinator_Plant_Interactions_taxonKeyFixed.csv")
ixn.reset_index(drop=True)
ixn.to_csv("Pollinator_Plant_Interactions_taxonKeyFixed.csv", index=False)
print("Index removed")

# hasher = FeatureHasher(input_type='string')
# X = hasher.transform(package(g) for g in ixn.pollinator_taxonKey)
# print("Hasher found ", n_nonzero_columns(X))

ixn.drop(columns=["plant_country", "pollinator_country"], inplace=True)
ixn.drop_duplicates(inplace=True)

X = ixn.drop(columns=["class"])
print("Binary Encoding for pollinator species")
bin = ce.BinaryEncoder().fit_transform(X.pollinator_species)
print(bin.head())
print(bin.shape)

print("Binary Encoding for whole kit and kaboodle")
X = X.drop(["pollinator_taxonKey", "plant_taxonKey", "pollinator_scientific", "plant_scientific"])
bin2 = ce.BinaryEncoder().fit_transform(X)
print(bin2.head())
print(bin2.shape)

breakpoint()

n_orig_features = ixn.shape[1]
ct = ColumnTransformer([(f't_{i}', FeatureHasher(n_features=2 ** 12,
                                                 input_type='string'), i) for i in range(n_orig_features)])

res = ct.fit_transform(ixn)
print(res.shape)
print(res)

# X2 = hasher.transform(ixn_pol_genus.to_dict(orient='list'))
# print("List hasher found", n_nonzero_columns(X2))
