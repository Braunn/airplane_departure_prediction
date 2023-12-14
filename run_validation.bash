#!/bin/bash

#SBATCH --job-name=merge_validation_run                 # sets the job name
#SBATCH --nodes=1                                 # reserves 5 machines
#SBATCH --mem=100Gb                               # reserves 100 GB memory
#SBATCH --partition=courses                       # requests that the job is executed in partition my partition
#SBATCH --time=3:00:00                            # reserves machines/cores for 4 hours.
#SBATCH --output=data/output/run_3/val_run_small.%j.out               # sets the standard output to be stored in file my_nice_job.%j.out, where %j is the job id)
#SBATCH --error=data/output/run_3/val_run_small.%j.err                # sets the standard error to be stored in file my_nice_job.%j.err, where %j is the job id)

RUN_NUM = 3

DATA_SET_PATH="/scratch/augenbraun.n/airplane_departure_prediction/data/processed/data_set_final"
echo $DATA_SET_PATH

lscpu > data/output/run_$RUN_NUM/cpu_info_validation_run.txt

# RIDGE REGRESSION MODEL
for lam in 0.0001 0.001;
do spark-submit validateModel.py --master local[35] --N 35 \
                              --regParam $lam \
                              --data_set $DATA_SET_PATH \
                              --K 4 --type RIDGE_REG --seed 5 --intercept 1 
done

# LASSO REGRESSION MODEL
for lam in 0.0001 0.001;
do spark-submit validateModel.py --master local[35] --N 35 \
                              --regParam $lam \
                              --data_set $DATA_SET_PATH \
                              --K 4 --type LASSO_REG --seed 5 --intercept 1 
done


