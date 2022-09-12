#!/bin/bash

cd ../processed_data
tar cvzf processed_data.tar.gz *.csv

cd ../source_data
tar cvzf source_data.tar.gz *.csv
