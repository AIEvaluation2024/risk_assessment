#!/bin/bash

alias activate=". ~/virtual_env/bin/activate"

declare -a datasets=("Kin8" "California_Housing" "Naval_Propulsion" "CCPP" "WineWhite")

# running the split method

for i in "${datasets[@]}"
do
	   python run_RA_average.py "NN" $i 
	   python run_RA_average.py "MVE_NN" $i
	      # or do whatever with individual element of the array
done

echo "Finished"







