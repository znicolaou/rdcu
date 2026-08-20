#!/bin/bash
#SBATCH --account=isaac-utk0437
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=00-03:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --export=all
#SBATCH --output=outs/sweep2_%a.out # where STDOUT goes
#SBATCH --error=outs/sweep2_%a.err # where STDERR goes
#SBATCH --array=1-11
module load cuda


i=$((SLURM_ARRAY_TASK_ID/11+1))
j=$((SLURM_ARRAY_TASK_ID%11+1))
#for j in `seq 1 11`; do
	js=`jobs -r | wc -l`
	sleep 5
	while [ $js -ge 1 ]; do 
		js=`jobs -r | wc -l`
		#echo -en "$js \r"
		#jobs
		sleep 10
	done
	#if [ ! -f scratch/3dcgle/${i}/${j}/faces.dat ]; then 
		echo "./makelines.py --states 0 --threads 32 --T 2000 --dt 0.1 --T1 1900 --dt1 10 --rm 1 --Ns 128 128 128 --Ls 200 200 200 --filebase scratch/3dcgle/${i}/${j}/ &"
		./makelines.py --states 0 --threads 32 --T 2000 --dt 0.1 --T1 1900 --dt1 10 --rm 1 --Ns 128 128 128 --Ls 200 200 200 --filebase scratch/3dcgle/${i}/${j}/ &
	#fi
#done
wait
