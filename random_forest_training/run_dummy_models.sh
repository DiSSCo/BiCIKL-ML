#!/bin/bash

# Control: Data are not stratified
nohup python3 RandomForestDummy.py -n Mod_0 -CT T > mod0.out &

# Data are stratified
nohup python3 RandomForestDummy.py -n Mod_1_dum > mod1.out &

# Model 2: Weights balanced
nohup python3 RandomForestDummy.py -n Mod_2_WB_dum -WB T > mod2.out &

# Model 3: Geo Training included
nohup python3 RandomForestDummy.py -n Mod_3_GT_dum -GT T > mod3.out &

# Model 4: Using Hashed Data
nohup python3 RandomForestDummy.py -n Mod_4_GT_dum -HD T > mod4.out &

# Model 5: Using SMOTE
nohup python3 RandomForestDummy.py -n Mod_5_GT_dum -SM T > mod5.out &


# get bounding box around occurence points (coords)ls
