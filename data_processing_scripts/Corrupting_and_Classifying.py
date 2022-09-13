import hashlib
import math
import os.path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings


def hash_str(s):
    return int.from_bytes(hashlib.md5(s.encode()).digest(), "big")


rewrite = False  # Set this to true if you want to re-do the corrupted data
pd.options.mode.chained_assignment = None

pwd = os.path.dirname(__file__)
source_data_path = os.path.join(pwd, os.path.relpath("../source_data/PollPlantsGBIF2.csv", pwd))

PollPlants = pd.read_csv(source_data_path)
PollPlants = PollPlants.dropna()

# Split plants and polls
subset = (PollPlants.iloc[:, 0:20])

# Drop scientific name and taxonKey for both pollinators and plants
# Scientific name is just a combination of genus and species; does not provide additional info
# TaxonKey is (essentially) randomly assigned; does not provide useful information

subset.drop(columns=['pollinator_scientificName', 'plant_scientificName', 'pollinator_scientific', 'plant_scientific'], inplace=True)
subset.drop(columns=['pollinator_taxonKey', 'plant_taxonKey'], inplace=True)
subset.drop_duplicates(inplace=True)
subset.dropna(inplace=True)

# Drop country information. This will be used to check that a pollinator-plant pair is valid
subset_no_geo = subset.drop(columns=['pollinator_country', 'plant_country'])
subset_no_geo.drop_duplicates(inplace=True)

# We drop country here because this is just shuffling data.
subset.drop_duplicates(inplace=True)

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
subset_no_geo.reset_index(drop=True, inplace=True)

# Number of unique pollinators and plants
poll_num = polls.shape[0]
plant_num = plants.shape[0]

# Path info
processed_data_path = "../processed_data"
processed_data_path = os.path.join(pwd, os.path.relpath(processed_data_path, pwd))
corrupted_path = processed_data_path + "/Corrupted_PollPlant.csv"  # Where we'll save our corrupted data

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
        plant_poll_combined = pd.merge(plant_poll_combined, subset_no_geo, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge',axis=1)

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

# Map categorical values to integers
for column in X.columns:
    str_to_int = {value: i for i, value in enumerate(sorted(list(X[column].unique())))}
    # create column with ints
    X[column + "_int"] = X[column].apply(lambda val: str_to_int[val])
    del X[column]  # remove original column (with strings)

for column in X_hashed.columns:
    # create column with md5 hash
    X_hashed[column + "_hash"] = X_hashed[column].apply(lambda val: hash_str(val))
    del X_hashed[column]  # remove original column (with strings)

# Save work to CSV
subset.to_csv((processed_data_path + "/Pollinator_Plant_Interactions.csv"), index=False)
X_hashed.to_csv((processed_data_path + "/Pollinator_Plant_hash_mapped.csv"), index=False)
X.to_csv((processed_data_path + "/Pollinator_Plant_int_mapped.csv"), index=False)
y.to_csv((processed_data_path + "/classes.csv"), index=False)



