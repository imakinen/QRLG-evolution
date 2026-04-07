#!/bin/bash

# Euclidean model
while IFS= read -r line; do
	python main.py $line
	julia evolution.jl $line
done < args_euclidean.txt

# Lorentzian model
while IFS= read -r line; do
	julia evolution.jl $line
done < args_lorentzian.txt

# Parametrized toy model
while IFS= read -r line; do
	python main.py $line
	julia evolution.jl $line
done < args_abc.txt
