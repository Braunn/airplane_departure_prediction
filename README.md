## AIRPORT DEPARTURE DELAY PREDICTION

# Validating Model
usage: validateModel.py [-h] [--master MASTER] [--N N] [--iter ITER]
                        [--regParam REGPARAM] [--data_set DATA_SET] [--K K]
                        [--type {LIN_REG,LASSO_REG,RIDGE_REG}] [--seed SEED]
                        [--intercept INTERCEPT]

Model Validation for Airplane Departure Prediction

optional arguments:
  -h, --help            show this help message and exit
  --master MASTER       Spark Master (default: local[30])
  --N N                 Number of partitions to be used in RDDs containing
                        departure and/or weather data (default: 30)
  --iter ITER           Number of iterations to use for training (default:
                        10000)
  --regParam REGPARAM   The regularization parameter to use for lasso/ridge
                        regression (default: 0.1)
  --data_set DATA_SET   Path to the data set (default: None)
  --K K                 Number of folds in k-fold cross validation (default:
                        4)
  --type {LIN_REG,LASSO_REG,RIDGE_REG}
                        Type of model for validating. All models train using
                        SGD (default: None)
  --seed SEED           Random seed used to shuffle the data set (-1 will use
                        a randomly generated seed based on time) (default: -1)
  --intercept INTERCEPT
                        Flag to add bias term in features (default: True)

Example:

spark-submit validateModel.py --master local[30] --N 30\
                              --iter 10 --regParam 0\
                              --data_set data/processed/data_set_2_departure\
                              --K 4 --type LIN_REG --seed 5 --intercept 1

# CONDA ENVIRONMENT SETUP 
clone the EECE5645 environment:
conda create --name EECE5645_clone --clone /courses/EECE5645.202410/data/bin/conda/EECE5645/ 

add sklearn pakage: 
conda install -c anaconda scikit-learn
conda install sklearn jupyter

# ENVIRONMENT SOURCING 
run:
source loadenv

# AUTH ISSUES WITH GITHUB
generating pub key on cluster
https://stackoverflow.com/questions/19660744/git-push-permission-denied-public-key
adding to github 
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account