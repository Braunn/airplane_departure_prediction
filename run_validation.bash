#!/bin/bash

lscpu > cpu_info_validation_run.txt

# LASSO REGRESSION MODEL
for lam in 0.01 0.1 1 100 1000 10000;
do spark-submit validateModel.py --master local[30] --N 30 \
                              --regParam $lam \
                              --data_set /scratch/augenbraun.n/airplane_departure_prediction/data/processed/data_set \
                              --K 4 --type LASSO_REG --seed 5 --intercept 1
done

# RIDGE REGRESSION MODEL
for lam in 0.01 0.1 1 100 1000 10000;
do spark-submit validateModel.py --master local[30] --N 30 \
                              --regParam $lam \
                              --data_set /scratch/augenbraun.n/airplane_departure_prediction/data/processed/data_set \
                              --K 4 --type RIDGE_REG --seed 5 --intercept 1
done

# LINEAR REGRESSION MODEL
spark-submit validateModel.py --master local[30] --N 30 \
                              --regParam 0 \
                              --data_set /scratch/augenbraun.n/airplane_departure_prediction/data/processed/data_set \
                              --K 4 --type LIN_REG --seed 5 --intercept 1

spark-submit validateModel.py --master local[30] --N 30 \
                              --regParam 0 \
                              --data_set /scratch/augenbraun.n/airplane_departure_prediction/data/processed/data_set \
                              --K 4 --type LIN_REG --seed 5 --intercept 0
