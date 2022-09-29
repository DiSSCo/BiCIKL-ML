#!/bin/bash

# Control: Data are not stratified
nohup python3 RandomForest.py -n Mod_0 -CT T > mod0.out &

# Data are stratified
nohup python3 RandomForest.py -n Mod_1 > mod1.out &

# Model 2: Weights balanced
nohup python3 RandomForest.py -n Mod_2_WB -WB T > mod2.out &

# Model 3: Geo Training included
nohup python3 RandomForest.py -n Mod_3_GT -GT T > mod3.out &

# Model 4: Using Hashed Data
nohup python3 RandomForest.py -n Mod_4_GT -HD T > mod4.out &

# Model 5: Using SMOTE
nohup python3 RandomForest.py -n Mod_5_GT -SM T > mod5.out &
