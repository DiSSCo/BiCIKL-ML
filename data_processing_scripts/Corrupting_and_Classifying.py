import hashlib
import json
import math
import os.path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings


def hash_str(st):
    hash_list = []
    for s in st:
        hash_list.append(int.from_bytes(hashlib.md5(s.encode()).digest(), "big"))
    return hash_list

def get_taxon_level(p):

    levels = ["kingdom", "phylum", "order", "family", "genus", "species"]
    for l in levels:
        if l in p: return l

    print("taxon level not found. column name: ", p)
    return "other"


def getSpecies(sci_name): # Get species name from scientific name (i.e. "genus species")
    try:
        sp_name = sci_name.split()[1]
    except IndexError:
        return sci_name
    return sp_name


strict = False
series_name = "SMOTE_Model"
if strict:
    series_name = series_name + "_Strict"


rewrite = True# Set this to true if you want to re-do the corrupted data
pd.options.mode.chained_assignment = None

pwd = os.path.dirname(__file__)
source_data_path = os.path.join(pwd, os.path.relpath("../source_data/Poll_Plant_Smote_Data.csv", pwd))

PollPlants = pd.read_csv(source_data_path, encoding='cp1252')

# Drop scientific name and taxonKey for both pollinators and plants
# Scientific name is <Genus Species Year>; does not provide additional/relevant info
# TaxonKey is (essentially) randomly assigned; does not provide useful information

subset = (PollPlants.drop(columns=["pollinator_taxonKey", "plant_taxonKey"]))
subset.drop_duplicates(inplace=True)
subset.dropna(inplace=True)

subset["pollinator_species"] = subset["pollinator_species"].apply(lambda tax: getSpecies(tax))
subset["plant_species"] = subset["plant_species"].apply(lambda tax: getSpecies(tax))


# Drop country information and save it to a separate DF.
# This will be used to check that a pollinator-plant pair is valid, even if their countries are different
# subset_no_geo = subset.drop(columns=['sourceCountry', 'targetCountry'])
# subset_no_geo.drop_duplicates(inplace=True)

# We drop country here because this is just shuffling data.
# subset.drop_duplicates(inplace=True)

# Midpoint to split plants from pollinators
col_mid = math.floor(subset.shape[1]/2)

# Split pollinators and plants
polls = subset.iloc[:, 0:col_mid]
polls.drop_duplicates(inplace=True)

plants = subset.iloc[:, col_mid:subset.shape[1]]
plants.drop_duplicates(inplace=True)

# Reset indices
polls.reset_index(drop=True, inplace=True)
plants.reset_index(drop=True, inplace=True)
subset.reset_index(drop=True, inplace=True)

#subset_no_geo.reset_index(drop=True, inplace=True)

# Number of unique pollinators and plants
poll_num = polls.shape[0]
plant_num = plants.shape[0]

# Path info
processed_data_path = "../processed_data/"
processed_data_path = os.path.join(pwd, os.path.relpath(processed_data_path, pwd))
corrupted_path = processed_data_path + series_name + "_corrupted_string.csv"  # Where we'll save our corrupted data

# Delete corrupted data file if it exists already
# Otherwise, we'll be appending to an existing file
if os.path.exists(corrupted_path) and rewrite:
    os.remove(corrupted_path)

if not os.path.exists(corrupted_path):
    for i in tqdm(range(plant_num)):
        # Repeat plant species for every pollinator species
        plant_long_df = pd.DataFrame(np.repeat(plants.iloc[[i]].values, poll_num, axis=0))

        plant_long_df.columns = plants.columns
        # Combine repeated plant list with each pollinator

        plant_poll_combined = pd.concat([polls, plant_long_df], ignore_index=True, axis=1)
        plant_poll_combined.columns = subset.columns

        # Drop whichever plant<->poll pairs are actually true (taxon only)
        plant_poll_combined = pd.merge(plant_poll_combined, subset, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge',axis=1)

        # We don't want to use our full "pair space" to train the classifier -> this could lead to overfitting
        # Randomly select 70% of 0s to be used in training
        if not strict:
            plant_poll_combined = plant_poll_combined.sample(frac=0.7)

        # Write to csv (we do this every round, otherwise the resulting dataframe grows too much per loop)
        # Funny enough, writing to disk each loop is faster than appending thousands of rows
        # Trust me
        plant_poll_combined.to_csv(corrupted_path, index=False, mode='a', header=not os.path.exists(corrupted_path))

corrupted = pd.read_csv(corrupted_path)  # Pull the corrupted data that we wrote to disk


# Set classes, rejoin dataframes
subset["class"] = 1
corrupted["class"] = 0

concatenated = pd.concat([subset, corrupted], ignore_index=True)

correct_df_nas = subset.isna().sum().sum()
corrupted_df_nas = corrupted.isna().sum().sum()
concatenated_df_nas = concatenated.isna().sum().sum()

# Check for NaNs
if corrupted_df_nas > 0:
    warnings.warn("Corrupted data has NaN values. An error has occurred in its generation")
if correct_df_nas > 0:
    warnings.warn("Original data has NaN values. Verify source data")
if concatenated_df_nas > 0:
    warnings.warn("Concatenated data has NaN values. An error in concatenation has occurred; check index alignment")

# Separate the columns into independent and dependent variables (or features and labels).
X = concatenated.drop(columns=["class"])
X_hashed = concatenated.drop(columns=["class"])

y = concatenated['class']  # Labels

#taxonomy_enum = pd.DataFrame(columns=["taxon_group", "group_id", "taxon_level", "relation_type"])
taxonomy_enum = {}

# Map categorical values to integers
for column in X.columns:
    str_to_int = {value: i for i, value in enumerate(sorted(list(X[column].unique())))}
    # create column with ints
    X[column + "_int"] = X[column].apply(lambda val: str_to_int[val])
    taxonomy_enum = {**taxonomy_enum, **str_to_int}
    del X[column]  # remove original column (with strings)

with open(processed_data_path + "/" + series_name + "_taxon_mapping", 'w+') as f:
    f.write(json.dumps(taxonomy_enum, indent=2))

'''
taxonomy_hash = {}
for column in X_hashed.columns:
    # create column with md5 hash
    str_to_hash = {value: i for i, value in hash_str(sorted(list(X[column].unique())))}

    X_hashed[column + "_hash"] = X_hashed[column].apply(lambda val: str_to_hash(val))
    
    taxonomy_hash = {**taxonomy_hash, **str_to_hash}
    del X_hashed[column]  # remove original column (with strings)
    
    '''

# Save work to CSV
subset.to_csv((processed_data_path + "/" + series_name + "_subsetted_interactions.csv"), index=False)
# X_hashed.to_csv((processed_data_path + "/Pollinator_Plant_hash_mapped.csv"), index=False)
X.to_csv((processed_data_path + "/" + series_name + "_int_mapping.csv"), index=False)
y.to_csv((processed_data_path + "/" + series_name + "_classes.csv"), index=False)



