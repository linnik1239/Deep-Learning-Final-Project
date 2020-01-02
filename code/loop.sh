#!/bin/bash

Optimizer=("adam" "sgd")
LearningRate=(0.01)
Dropout=(0.3 0.5)
BatchSize=(11 22 44 121)
Augmentation=(11 22 44 121)
Division=(1 8 16 32)
Blank=(0 1) # 0-without 1-with
Models=("simpleConv" "smallUNet" "smallUNet_less" "UNet")

# get length of an arrays
opt_len=${#Optimizer[@]}
lr_len=${#LearningRate[@]}
drop_len=${#Dropout[@]}
batch_len=${#BatchSize[@]}
augmentation_len=${#Augmentation[@]}
division_len=${#Division[@]}
blank_len=${#Blank[@]}
model_len=${#Models[@]}

# use for loop read all nameservers
counter=$((4))
for (( i=0; i<${opt_len}; i++ )); do
	for (( j=0; j<${lr_len}; j++ )); do
		for (( k=0; k<${drop_len}; k++ )); do
			for (( m=0; m<${batch_len}; m++ )); do
				for (( n=0; n<${augmentation_len}; n++ )); do
					for (( p=0; p<${division_len}; p++ )); do
						for (( q=0; q<${blank_len}; q++ )); do
							for (( l=0; l<${model_len}; l++ )); do
								counter=$((counter+1))
								echo "#PBS -N train$counter" > launch$counter
								echo "cd /home/u24913/Tammy/" >> launch$counter
								echo "./time python main.py ${Optimizer[$i]} ${LearningRate[$j]} ${Dropout[$k]} ${BatchSize[$m]} ${Augmentation[$n]} ${Division[$p]} ${Blank[$q]} ${Models[$l]} $counter" | tr "\n" " " >> launch$counter
								echo "" >> launch$counter
								qsub launch$counter
								time=`echo $line | cut -d" " -f 3 | cut -d"e" -f 1`
								cpu=`echo $line | cut -d" " -f 4 | cut -d"C" -f 1`
								mem=`echo $line | cut -d" " -f 6 | cut -d"m" -f 1`
								echo "$counter $time $cpu $mem"
							done
						done
					done
				done
			done
		done
	done
done



