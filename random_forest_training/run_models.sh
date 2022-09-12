#!/bin/bash

# Control: data are not stratified
python3 RandomForest.py -n Mod_0 -CT T

# Data are stratified, no other variables
python3 RandomForest.py -n Mod_1

# Model 2: Weights balanced
python3 RandomForest.py -n Mod_2_WB -WB T

# Model 3: Geo Training included
python3 RandomForest.py -n Mod_3_GT -GT T

# Model 4: Using Hashed Data
python3 RandomForest.py -n Mod_4_GT -HD T

# Model 5: Using SMOTE
python3 RandomForest.py -n Mod_5_GT -SM T




