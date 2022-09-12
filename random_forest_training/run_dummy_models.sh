#!/bin/bash

# Control: Data are not stratified
python3 RandomForest.py -n Mod_0 -CT T

# Data are stratified
python3 RandomForestDummy.py -n Mod_1_dum

# Model 2: Weights balanced
python3 RandomForestDummy.py -n Mod_2_WB_dum -WB T

# Model 3: Geo Training included
python3 RandomForestDummy.py -n Mod_3_GT_dum -GT T

# Model 4: Using Hashed Data
python3 RandomForestDummy.py -n Mod_4_GT_dum -HD T

# Model 5: Using SMOTE
python3 RandomForestDummy.py -n Mod_5_GT_dum -SM T



# get bounding box around occurence points (coords)
