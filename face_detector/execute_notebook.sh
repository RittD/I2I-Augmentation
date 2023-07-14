#!/bin/bash

echo " "
echo "Converting notebook: $1"
jupyter nbconvert --to python $1

nb_name=$(echo $1 | cut -d "." -f 1)

echo " "
echo "Running script: $nb_name.py"
python3.7 $nb_name.py
