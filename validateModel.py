import argparse
import numpy as np

from pyspark import SparkContext
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LassoWithSGD
from pyspark.mllib.regression import RidgeRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics

from datetime import datetime

from runRDD import readRDD, debugPrint, swap
from trainModel import createFeatureVector, printRegressionMetrics

def crossValidation(trainRDD, folds, model_name, iter, intercept, reg=0, seed=-1):
    '''
    Preforms k-folds cross validation on the trainRDD
    Input:
        trainRDD : RDD with the training set 
        folds : number of partitions to validate on
        model_name : (string) 
    '''
    
    # set random seed for shuffling data
    if seed == -1:
        seed = np.random.randint(100000)
    rng = np.random.RandomState(seed)
    print(f'Using seed {seed} to shuffle trainRDD\n')

    # shuffle data by changing their indexes
    nTraining = trainRDD.count()
    shuffle_map = rng.permutation(np.arange(nTraining))
    trainRDD = trainRDD.zipWithIndex()\
        .map(swap)\
        .map(lambda pair: (shuffle_map[pair[0]], pair[1]))\
        .cache()
    
    # check that number of folds makes sense given the training set size
    if nTraining < folds:
        print(f'Not enough training data for the number of blocks requested')
        return -1
    
    blockSize = np.ceil(nTraining/folds)
    sizePerBlock = np.ones(folds)*blockSize
    sizePerBlock[-1] = sizePerBlock[-1] + (nTraining -sum(sizePerBlock))

    # start cross validation
    startTime = datetime.now()
    for k in range(folds):
        print(f'Starting fold {k}')
        foldStartTime = datetime.now()

        # get indexes for fold
        startIndex = sum(sizePerBlock[0:k])
        endIndex = startIndex + sizePerBlock[k]

        # debug
        print(f'Count = {nTraining}, start index {startIndex}, stop index: {endIndex}')

        # parition training set 
        trainingSet = trainRDD.filter(lambda pair : not (pair[0] >= startIndex and pair[0] < endIndex))\
                                .values()
        testSet = trainRDD.filter(lambda pair : pair[0] >= startIndex and pair[0] < endIndex)\
                                .values()
        
        # train model
        if model_name ==  'LIN_REG':
                trainedModel = LinearRegressionWithSGD.train(trainingSet, iterations=iter, intercept=intercept)
        elif model_name == 'LASSO_REG':
                trainedModel =            LassoWithSGD.train(trainingSet, iterations=iter, regParam=reg, intercept=intercept)
        elif model_name == 'RIDGE_REG':
                trainedModel =  RidgeRegressionWithSGD.train(trainingSet, iterations=iter, regParam=reg, intercept=intercept)
        else:
            print(f'Could not find model {model_name}')
            return -1

        # evaluate on training data
        predictRDD = testSet.map(lambda x: (float(trainedModel.predict(x.features)), float(x.label)) )
        print(f'Results for fold = {k} training using {model_name}\n')
        printRegressionMetrics(predictRDD, True)

        # print model parameters
        print(f"\tweights =\n\t",trainedModel.weights,"\n")
        print(f"\tintercept =",trainedModel.intercept,"\n")

        # print timing information 
        print(f'End of fold {k}\n')
        print(f'Duration of fold iteration {datetime.now() - foldStartTime}')
        print(f'Duration of fold iteration {datetime.now() - startTime}\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Model Validation for Airplane Departure Prediction',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--master',   type=str,   default="local[30]",                  help="Spark Master")
    parser.add_argument('--N',        type=int,   default=30,                           help="Number of partitions to be used in RDDs containing departure and/or weather data")
    parser.add_argument('--iter',     type=int,   default=10000,                        help="Number of iterations to use for training")
    parser.add_argument('--regParam', type=float, default=0.1,                          help="The regularization parameter to use for lasso/ridge regression")
    parser.add_argument('--data_set', type=str,                                         help="Path to the data set")
    parser.add_argument('--K',        type=int,   default=4,                            help="Number of folds in k-fold cross validation")
    parser.add_argument('--type',     choices=['LIN_REG', 'LASSO_REG', 'RIDGE_REG'],    help='Type of model for validating. All models train using SGD') 
    parser.add_argument('--seed',     type=int, default=-1,                           help="Random seed used to shuffle the data set (-1 will use a randomly generated seed based on time)")
    parser.add_argument('--intercept',type=bool,  default=True,                         help="Flag to add bias term in features")
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Model Validation Airplane Departure Prediction')
    sc.setLogLevel('warn')

    print("Command line arguments:")
    print("\t--master",args.master)
    print("\t--N",args.N)
    print("\t--iter",args.iter)
    print("\t--regParam",args.regParam)
    print("\t--data_set",args.data_set)
    print("\t--K",args.K)
    print("\t--type",args.type)
    print("\t--seed",args.seed)
    print("\t--intercept",args.intercept)
    print("\n")

    # Read in the RDD from the filesystem
    trainRDD = readRDD(sc, args.data_set)

    # Let's convert the RDD to LabeledPoint data types for training
    trainRDD = trainRDD.map(lambda x: LabeledPoint(x[0], createFeatureVector(x[1][2],x[1][3],x[1][4]))).cache()

    # Debug print out a few of them
    print("departureWeatherRDD.count() =",trainRDD.count(),"\n")
    print("departureWeatherRDD.takeSample(False, 5) =\n",trainRDD.takeSample(False, 5),"\n")

    print(datetime.now(),"- Training linear regression model with stochastic gradient descent")

    # let it rip!
    crossValidation(trainRDD, args.K, args.type, args.iter, args.intercept, args.regParam, args.seed)    

    print(datetime.now(),"- C'est fini")
