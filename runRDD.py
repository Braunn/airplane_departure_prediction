import argparse
import numpy as np

from pyspark import SparkContext
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LassoWithSGD
from pyspark.mllib.regression import RidgeRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics

from datetime import datetime

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Airplane Departure Prediction',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--master',default="local[30]",help="Spark Master")
    parser.add_argument('--N',type=int,default=30,help="Number of partitions to be used in RDDs containing departure and/or weather data.")
    parser.add_argument('--split',default=0.8,help="Percentage of data to split for training vs test")
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Airplane Departure Prediction')
    sc.setLogLevel('warn')

    print("Command line arguments:")
    print("\t--master",args.master)
    print("\t--N",args.N)
    print("\t--split",args.split)
    print("\n")

    # Read in the airport departure statistics from CSV in to an RDD and split by commas
    departureRDD = sc.textFile("concatenated_data.csv").map(lambda x: x.split(","))

    # Strip the RDD down to the columns we care about: (origin airport, (scheduled departure date, scheduled departure time, weather delay in minutes))
    departureRDD = departureRDD.map(lambda x: (x[17], (x[1], x[5], float(x[13]))) ).repartition(args.N)

    # Combine the date and time in to one string
    departureRDD = departureRDD.mapValues(lambda x: (x[0] + " " + x[1], x[2]) )

    # Convert that date/time string to a datetime object
    departureRDD = departureRDD.mapValues(lambda x: (datetime.strptime(x[0], "%m/%d/%Y %H:%M"), x[1]) )

    # Debug print out a few of them
    print("departureRDD.count() =",departureRDD.count(),"\n")
    print("departureRDD.takeSample(False, 5) =\n\n",departureRDD.takeSample(False, 5),"\n")

    # Read in the weather data from CSV in to an RDD and split by commas
    weatherRDD = sc.textFile("cleaned_weather_data.csv").map(lambda x: x.split(","))

    # Strip the RDD down to the columns we care about: (AirportCode,(Type,Severity,StartTime(UTC),EndTime(UTC),Precipitation(in)))
    weatherRDD = weatherRDD.map(lambda x: (x[7],(x[1],x[2],x[3],x[4],float(x[5])))).repartition(args.N)

    # Convert that date/time string to a datetime object
    weatherRDD = weatherRDD.mapValues(lambda x: (x[0], x[1], datetime.strptime(x[2], "%Y-%m-%d %H:%M:%S"), datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), x[4]) )

    # Debug print out a few of them
    print("weatherRDD.count() =",weatherRDD.count(),"\n")
    print("weatherRDD.takeSample(False, 5) =\n\n",weatherRDD.takeSample(False, 5),"\n")

    # Join the two RDDs together
    departureWeatherRDD = departureRDD.join(weatherRDD,args.N)

    # Filter down the RDD to entries where scheduled departure time (x[1][0][0]) is between weather start time (x[1][1][2]) and weather stop time (x[1][1][3])
    departureWeatherRDD = departureWeatherRDD.filter(lambda x: x[1][1][2] <= x[1][0][0] and x[1][0][0] <= x[1][1][3])

    # Clean up the value part of the tuple so it's easier to index in to
    departureWeatherRDD = departureWeatherRDD.mapValues(lambda x: (x[0][0], x[0][1], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4]))

    # Now the RDD consists of key/value pairs where the values are ordered as:
    #     scheduled departure time (datetime)
    #     weather delay in minutes (float)
    #     the weather event category ('rain',etc)
    #     the weather event severity ('light','moderate','severe','heavy')
    #     start time of the weather event (datetime)
    #     stop time of the weather event (datetime)
    #     precipitation in inches (float)

    # TODO: We need to take the weather event category and severity and one hot encode them. We'll also need to artificially add an enumerated value to both of them for "none" in case there were no weather events going on at that time.

    # Let's convert the RDD to LabeledPoint data types for training
    departureWeatherRDD = departureWeatherRDD.mapValues(lambda x: LabeledPoint(x[1],np.array([x[6],1.0]))) # TODO: Get rid of this dummy 1.0 I added

    # Cache it
    departureWeatherRDD = departureWeatherRDD.cache()

    # Debug print out a few of them
    print("departureWeatherRDD.count() =",departureWeatherRDD.count(),"\n")
    print("departureWeatherRDD.takeSample(False, 5) =\n\n",departureWeatherRDD.takeSample(False, 5),"\n")

    # Let's create a training set and test set
    trainingRDD,testRDD = departureWeatherRDD.randomSplit([args.split,1.0-args.split])

    print("trainingRDD.count() = ", trainingRDD.count())
    print("testRDD.count() = ", testRDD.count())

    linearRegModel = LinearRegressionWithSGD.train(trainingRDD.values(), iterations=100)

    predictRDD = testRDD.mapValues(lambda x: (float(linearRegModel.predict(x.features)), float(x.label)) )

    # Compute the metrics for the linear regression model using the subset of data we've set aside for testing
    metrics = RegressionMetrics(predictRDD.values())

    print("\n\n*** Metrics for Linear Regression with Stochastic Gradient Descent ***")
    print("Explained Variance =",metrics.explainedVariance)
    print("Mean Absolute Error =",metrics.meanAbsoluteError)
    print("Mean Squared Error =",metrics.meanSquaredError)
    print("Root Mean Squared Error =",metrics.rootMeanSquaredError)
    print("R^2 =",metrics.r2)

    lassoRegModel = LassoWithSGD.train(trainingRDD.values(), iterations=100, regParam=0.001)

    predictRDD = testRDD.mapValues(lambda x: (float(lassoRegModel.predict(x.features)), float(x.label)) )

    # Compute the metrics for the lasso regression model using the subset of data we've set aside for testing
    metrics = RegressionMetrics(predictRDD.values())

    print("\n\n*** Metrics for Lasso Regression with Stochastic Gradient Descent ***")
    print("Explained Variance =",metrics.explainedVariance)
    print("Mean Absolute Error =",metrics.meanAbsoluteError)
    print("Mean Squared Error =",metrics.meanSquaredError)
    print("Root Mean Squared Error =",metrics.rootMeanSquaredError)
    print("R^2 =",metrics.r2)

    ridgeRegModel = RidgeRegressionWithSGD.train(trainingRDD.values(), iterations=100, regParam=0.001)

    predictRDD = testRDD.mapValues(lambda x: (float(ridgeRegModel.predict(x.features)), float(x.label)) )

    # Compute the metrics for the lasso regression model using the subset of data we've set aside for testing
    metrics = RegressionMetrics(predictRDD.values())

    print("\n\n*** Metrics for Ridge Regression with Stochastic Gradient Descent ***")
    print("Explained Variance =",metrics.explainedVariance)
    print("Mean Absolute Error =",metrics.meanAbsoluteError)
    print("Mean Squared Error =",metrics.meanSquaredError)
    print("Root Mean Squared Error =",metrics.rootMeanSquaredError)
    print("R^2 =",metrics.r2)
