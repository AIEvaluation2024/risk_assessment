#!/bin/bash

alias activate=". ~/virtual_env/bin/activate"

#declare -a datasets=("Kin8" "California_Housing" "Naval_Propulsion" "CCPP" "WineWhite")

declare -a datasets=("Exponential_Function")
# running the split method

for i in "${datasets[@]}"
do
		 python train_split.py "NN" $i
	   python train_split.py "MVE_NN" $i 
	   python train_CV.py "MVE_NN" $i
		 python train_CV.py "NN" $i
	      # or do whatever with individual element of the array
done

echo "Finished"







