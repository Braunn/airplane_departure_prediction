import argparse
import numpy as np

from pyspark import SparkContext
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LassoWithSGD
from pyspark.mllib.regression import RidgeRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics

from datetime import datetime

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
    parser.add_argument('--master', type=str,   default="local[30]", help="Spark Master")
    parser.add_argument('--N',      type=int,   default=30,          help="Number of partitions to be used in RDDs containing departure and/or weather data.")
    parser.add_argument('--split',  type=float, default=0.8,         help="Percentage of data to split for training vs test")
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
    print("departureRDD.takeSample(False, 5) =\n",departureRDD.takeSample(False, 5),"\n")

    # Read in the weather data from CSV in to an RDD and split by commas
    weatherRDD = sc.textFile("cleaned_weather_data.csv").map(lambda x: x.split(","))

    # Strip the RDD down to the columns we care about: (AirportCode,(Type,Severity,StartTime(UTC),EndTime(UTC),Precipitation(in)))
    weatherRDD = weatherRDD.map(lambda x: (x[7],(x[1],x[2],x[3],x[4],float(x[5])))).repartition(args.N)

    # Convert that date/time string to a datetime object
    weatherRDD = weatherRDD.mapValues(lambda x: (x[0], x[1], datetime.strptime(x[2], "%Y-%m-%d %H:%M:%S"), datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), x[4]) )

    # Debug print out a few of them
    print("weatherRDD.count() =",weatherRDD.count(),"\n")
    print("weatherRDD.takeSample(False, 5) =\n",weatherRDD.takeSample(False, 5),"\n")

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
    departureWeatherRDD = departureWeatherRDD.mapValues(lambda x: LabeledPoint(x[1], createFeatureVector(x[2],x[3],x[6])))

    # Cache it
    departureWeatherRDD = departureWeatherRDD.cache()

    # Debug print out a few of them
    print("departureWeatherRDD.count() =",departureWeatherRDD.count(),"\n")
    print("departureWeatherRDD.takeSample(False, 5) =\n",departureWeatherRDD.takeSample(False, 5),"\n")

    # Let's create a training set and test set
    trainingRDD,testRDD = departureWeatherRDD.randomSplit([args.split,1.0-args.split])

    print("trainingRDD.count() = ", trainingRDD.count())
    print("testRDD.count() = ", testRDD.count())

    linearRegModel = LinearRegressionWithSGD.train(trainingRDD.values(), iterations=100)

    predictRDD = testRDD.mapValues(lambda x: (float(linearRegModel.predict(x.features)), float(x.label)) )

    # Compute the metrics for the linear regression model using the subset of data we've set aside for testing
    metrics = RegressionMetrics(predictRDD.values())

    print("\n*** Metrics for Linear Regression with Stochastic Gradient Descent ***")
    print("Explained Variance =",metrics.explainedVariance)
    print("Mean Absolute Error =",metrics.meanAbsoluteError)
    print("Mean Squared Error =",metrics.meanSquaredError)
    print("Root Mean Squared Error =",metrics.rootMeanSquaredError)
    print("R^2 =",metrics.r2,"\n")

    lassoRegModel = LassoWithSGD.train(trainingRDD.values(), iterations=100, regParam=0.001)

    predictRDD = testRDD.mapValues(lambda x: (float(lassoRegModel.predict(x.features)), float(x.label)) )

    # Compute the metrics for the lasso regression model using the subset of data we've set aside for testing
    metrics = RegressionMetrics(predictRDD.values())

    print("\n*** Metrics for Lasso Regression with Stochastic Gradient Descent ***")
    print("Explained Variance =",metrics.explainedVariance)
    print("Mean Absolute Error =",metrics.meanAbsoluteError)
    print("Mean Squared Error =",metrics.meanSquaredError)
    print("Root Mean Squared Error =",metrics.rootMeanSquaredError)
    print("R^2 =",metrics.r2,"\n")

    ridgeRegModel = RidgeRegressionWithSGD.train(trainingRDD.values(), iterations=100, regParam=0.001)

    predictRDD = testRDD.mapValues(lambda x: (float(ridgeRegModel.predict(x.features)), float(x.label)) )

    # Compute the metrics for the lasso regression model using the subset of data we've set aside for testing
    metrics = RegressionMetrics(predictRDD.values())

    print("\n*** Metrics for Ridge Regression with Stochastic Gradient Descent ***")
    print("Explained Variance =",metrics.explainedVariance)
    print("Mean Absolute Error =",metrics.meanAbsoluteError)
    print("Mean Squared Error =",metrics.meanSquaredError)
    print("Root Mean Squared Error =",metrics.rootMeanSquaredError)
    print("R^2 =",metrics.r2,"\n")
