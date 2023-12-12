#!/bin/bash

DATA_SET_PATH="/scratch/augenbraun.n/airplane_departure_prediction/data/processed/data_set"
echo $DATA_SET_PATH

lscpu > cpu_info_validation_run.txt

# LASSO REGRESSION MODEL
for lam in 0.01 0.1 1 100 1000 10000;
do spark-submit validateModel.py --master local[30] --N 30 \
                              --regParam $lam \
                              --data_set $DATA_SET_PATH \
                              --K 4 --type LASSO_REG --seed 5 --intercept 1 >> lasso_regression_output.txt
done

# RIDGE REGRESSION MODEL
for lam in 0.01 0.1 1 100 1000 10000;
do spark-submit validateModel.py --master local[30] --N 30 \
                              --regParam $lam \
                              --data_set $DATA_SET_PATH \
                              --K 4 --type RIDGE_REG --seed 5 --intercept 1 >> ridge_regression_output.txt
done

# LINEAR REGRESSION MODEL
spark-submit validateModel.py --master local[30] --N 30 \
                              --regParam 0 \
                              --data_set $DATA_SET_PATH \
                              --K 4 --type LIN_REG --seed 5 --intercept 1 >> linear_regression_w_intercept.txt

spark-submit validateModel.py --master local[30] --N 30 \
                              --regParam 0 \
                              --data_set $DATA_SET_PATH \
                              --K 4 --type LIN_REG --seed 5 --intercept 0 >> linear_regression_wo_intercept.txt
