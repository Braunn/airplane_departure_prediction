#!/bin/bash

#SBATCH --job-name=big_merge                 # sets the job name
#SBATCH --nodes=1                                 # reserves 5 machines
#SBATCH --mem=100Gb                               # reserves 100 GB memory
#SBATCH --partition=courses                       # requests that the job is executed in partition my partition
#SBATCH --time=8:00:00                            # reserves machines/cores for 4 hours.
#SBATCH --output=data/output/data_set_combine.%j.out               # sets the standard output to be stored in file my_nice_job.%j.out, where %j is the job id)
#SBATCH --error=data/output/data_set_combine.%j.err                # sets the standard error to be stored in file my_nice_job.%j.err, where %j is the job id)

spark-submit runRDD.py