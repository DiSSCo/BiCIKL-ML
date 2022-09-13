#!/bin/bash

# Control: Data are not stratified
nohup python3 RandomForest.py -n Mod_0 -CT T > mod0.out &
echo "model 0 finished"

# Data are stratified
nohup python3 RandomForest.py -n Mod_1_dum > mod1.out &
echo "model 1 finished"

# Model 2: Weights balanced
nohup python3 RandomForest.py -n Mod_2_WB_dum -WB T > mod2.out &
echo "model 2 finished"

# Model 3: Geo Training included
nohup python3 RandomForest.py -n Mod_3_GT_dum -GT T > mod3.out &
echo "model 3 finished"

# Model 4: Using Hashed Data
nohup python3 RandomForest.py -n Mod_4_GT_dum -HD T > mod4.out &
echo "model 4 finished"

# Model 5: Using SMOTE
nohup python3 RandomForest.py -n Mod_5_GT_dum -SM T > mod5.out &
echo "model 5 finished"
