#!/bin/bash
#SBATCH --account=isaac-utk0437
#SBATCH --partition=ai-tenn-debug
#SBATCH --qos=ai-tenn-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --time=00-03:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --export=all
#SBATCH --output=outs/sweep_%a.out # where STDOUT goes
#SBATCH --error=outs/sweep_%a.err # where STDERR goes
#SBATCH --array=1-11
module load cuda


i=$((SLURM_ARRAY_TASK_ID))

gid=0
for j in `seq 1 11`; do
	b=`bc -l <<< "0.5+1.5*(${i}-1)/10"`
	c=`bc -l <<< "1.0/(${b}+0.5-0.75*(${j}-1)/10)"`
	mkdir -p scratch/3dcgle/${i}/${j}
	js=`jobs -r | wc -l`
	sleep 5
	while [ $js -ge 1 ]; do 
		js=`jobs -r | wc -l`
		#echo -en "$js \r"
		#jobs
		sleep 10
	done
	if [ ! -f scratch/3dcgle/${i}/${j}/states.dat ]; then 
		echo "./makelines.py --lines 0 --Ns 128 128 128 --Ls 200 200 200 --T 2000 --T1 1900 --dt1 10 --dt 0.1 --b ${b} --c ${c} --filebase scratch/3dcgle/${i}/${j}/ &"
		./makelines.py --lines 0 --Ns 128 128 128 --Ls 200 200 200 --T 2000 --T1 1900 --dt1 10 --dt 0.1 --b ${b} --c ${c} --filebase scratch/3dcgle/${i}/${j}/ &
	fi
	gid=$((gid+1))
	if [ $gid -ge 1 ]; then
		gid=0
	fi
done
wait
