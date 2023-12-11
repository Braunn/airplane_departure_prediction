import argparse
import numpy as np

from pyspark import SparkContext
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LassoWithSGD
from pyspark.mllib.regression import RidgeRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics

from datetime import datetime

from runRDD import readRDD

def createFeatureVector(category,severity,precipitation):
    """ Given a weather event category and severity (both are strings) and a floating point precipitation value,
        create the binarized feature vector

    Inputs are:
       - category (string): the weather event category which can be "Cold", "Fog", "Hail", "Precipitation", "Rain", "Snow", "Storm"
       - severity (string): the weather event severity which can be "Heavy", "Light", "Moderate", "Other", "Severe", "UNK"
       - precipitation (float): the precipitation in inches

    The return value is
        - a NumPy array where the indices represent:

         Weather Event Category

         Index 0 = Cold (true/false)
         Index 1 = Fog (true/false)
         Index 2 = Hail (true/false)
         Index 3 = Precipitation (true/false)
         Index 4 = Rain (true/false)
         Index 5 = Snow (true/false)
         Index 6 = Storm (true/false)

         Weather Event Severity

         Index 7 = Light (true/false)
         Index 8 = Moderate (true/false)
         Index 9 = Heavy (true/false)
         Index 10 = Severe (true/false)
         Index 11 = Other (true/false)
         Index 12 = Unknown (true/false)

         Miscellaneous

         Index 13 = Precipitation in inches (float)
    """
    features = np.zeros(14, dtype=float)

    if category == "Cold":
        features[0] = 1
    elif category == "Fog":
        features[1] = 1
    elif category == "Hail":
        features[2] = 1
    elif category == "Precipitation":
        features[3] = 1
    elif category == "Rain":
        features[4] = 1
    elif category == "Snow":
        features[5] = 1
    elif category == "Storm":
        features[6] = 1
    else:
        print("Got unexpected weather event category:",category)

    if severity == "Light":
        features[7] = 1
    elif severity == "Moderate":
        features[8] = 1
    elif severity == "Heavy":
        features[9] = 1
    elif severity == "Severe":
        features[10] = 1
    elif severity == "Other":
        features[11] = 1
    elif severity == "UNK":
        features[12] = 1
    else:
        print("Got unexpected weather event severity:",severity)

    features[13] = precipitation

    return features

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Airplane Departure Prediction',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--master',   type=str,   default="local[30]", help="Spark Master")
    parser.add_argument('--N',        type=int,   default=30,          help="Number of partitions to be used in RDDs containing departure and/or weather data")
    parser.add_argument('--split',    type=float, default=0.8,         help="Percentage of data to split for training vs test")
    parser.add_argument('--iter',     type=int,   default=10000,       help="Number of iterations to use for training")
    parser.add_argument('--regParam', type=float, default=0.1,         help="The regularization parameter to use for lasso/ridge regression")
    parser.add_argument('--data_set', type=str,   help="Path to the data set")
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Airplane Departure Prediction')
    sc.setLogLevel('warn')

    print("Command line arguments:")
    print("\t--master",args.master)
    print("\t--N",args.N)
    print("\t--split",args.split)
    print("\t--iter",args.iter)
    print("\t--regParam",args.regParam)
    print("\n")

    # Read in the RDD from the filesystem
    departureWeatherRDD = readRDD(sc, args.data_set)

    # Debug print out a few of them
    print("departureWeatherRDD.count() =",departureWeatherRDD.count(),"\n")
    print("departureWeatherRDD.takeSample(False, 5) =\n",departureWeatherRDD.takeSample(False, 5),"\n")

    # Let's convert the RDD to LabeledPoint data types for training
    departureWeatherRDD = departureWeatherRDD.map(lambda x: LabeledPoint(x[0], createFeatureVector(x[1][2],x[1][3],x[1][4])))

    # Cache it
    departureWeatherRDD = departureWeatherRDD.cache()

    # Debug print out a few of them
    print("departureWeatherRDD.count() =",departureWeatherRDD.count(),"\n")
    print("departureWeatherRDD.takeSample(False, 5) =\n",departureWeatherRDD.takeSample(False, 5),"\n")

    # Let's create a training set and test set
    trainingRDD,testRDD = departureWeatherRDD.randomSplit([args.split,1.0-args.split])

    print("trainingRDD.count() = ", trainingRDD.count())
    print("testRDD.count() = ", testRDD.count())

    linearRegModel = LinearRegressionWithSGD.train(trainingRDD, iterations=args.iter, intercept=True)

    print("Linear regression model weights =\n",linearRegModel.weights,"\n")
    print("Linear regression model intercept =",linearRegModel.intercept,"\n")

    predictRDD = testRDD.map(lambda x: (float(linearRegModel.predict(x.features)), float(x.label)) )

    # Compute the metrics for the linear regression model using the subset of data we've set aside for testing
    metrics = RegressionMetrics(predictRDD)

    print("\n*** Metrics for Linear Regression Model with Stochastic Gradient Descent ***")
    print("Explained Variance =",metrics.explainedVariance)
    print("Mean Absolute Error =",metrics.meanAbsoluteError)
    print("Mean Squared Error =",metrics.meanSquaredError)
    print("Root Mean Squared Error =",metrics.rootMeanSquaredError)
    print("R^2 =",metrics.r2,"\n")

    lassoRegModel = LassoWithSGD.train(trainingRDD, iterations=args.iter, regParam=args.regParam, intercept=True)

    print("Lasso regression model weights =\n",lassoRegModel.weights,"\n")
    print("Lasso regression model intercept =",lassoRegModel.intercept,"\n")

    predictRDD = testRDD.map(lambda x: (float(lassoRegModel.predict(x.features)), float(x.label)) )

    # Compute the metrics for the lasso regression model using the subset of data we've set aside for testing
    metrics = RegressionMetrics(predictRDD)

    print("\n*** Metrics for Lasso Regression Model with Stochastic Gradient Descent ***")
    print("Explained Variance =",metrics.explainedVariance)
    print("Mean Absolute Error =",metrics.meanAbsoluteError)
    print("Mean Squared Error =",metrics.meanSquaredError)
    print("Root Mean Squared Error =",metrics.rootMeanSquaredError)
    print("R^2 =",metrics.r2,"\n")

    ridgeRegModel = RidgeRegressionWithSGD.train(trainingRDD, iterations=args.iter, regParam=args.regParam, intercept=True)

    print("Ridge regression model weights =\n",ridgeRegModel.weights,"\n")
    print("Ridge regression model intercept =",ridgeRegModel.intercept,"\n")

    predictRDD = testRDD.map(lambda x: (float(ridgeRegModel.predict(x.features)), float(x.label)) )

    # Compute the metrics for the lasso regression model using the subset of data we've set aside for testing
    metrics = RegressionMetrics(predictRDD)

    print("\n*** Metrics for Ridge Regression Model with Stochastic Gradient Descent ***")
    print("Explained Variance =",metrics.explainedVariance)
    print("Mean Absolute Error =",metrics.meanAbsoluteError)
    print("Mean Squared Error =",metrics.meanSquaredError)
    print("Root Mean Squared Error =",metrics.rootMeanSquaredError)
    print("R^2 =",metrics.r2,"\n")
