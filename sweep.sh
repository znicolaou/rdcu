#!/bin/bash
#SBATCH --account=isaac-utk0437
#SBATCH --partition=ai-tenn-debug
#SBATCH --qos=ai-tenn-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00-03:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --export=all
#SBATCH --output=outs/sweep_%a.out # where STDOUT goes
#SBATCH --error=outs/sweep_%a.err # where STDERR goes
#SBATCH --array=1-10
module load cuda


j=$((SLURM_ARRAY_TASK_ID))

gid=0
seed0=$((SLURM_ARRAY_TASK_ID%10))
for i in `seq 1 10`; do
	b=`bc -l <<< "0.5+1.5*${i}/10"`
	c=`bc -l <<< "1.0/${b}-0.5+1.0*${j}/10"`
	echo $N $K $dK $seed
	mkdir -p data/3dcgle/${i}/${j}
	jobs=`jobs -r | wc -l`
	sleep 5
	while [ $jobs -ge 1 ]; do 
		jobs=`jobs -r | wc -l`
		#echo -en "$jobs \r"
		jobs -r
		sleep 100
	done
	if [ ! -f data/3dcgle/${i}/${j}/faces.dat ]; then 
		echo "./makelines.py --b ${b} --c ${c} --filebase data/3dcgle/${i}/${j}/ &"
		./makelines.py --b ${b} --c ${c} --filebase data/3dcgle/${i}/${j}/ &
	fi
	gid=$((gid+1))
	if [ $gid -ge 1 ]; then
		gid=0
	fi
done
wait
