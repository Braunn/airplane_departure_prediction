import argparse
import numpy as np

from pyspark import SparkContext
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LassoWithSGD
from pyspark.mllib.regression import RidgeRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics

from datetime import datetime

from runRDD import readRDD, debugPrint, swap, saveRDD
from trainModel import createFeatureVector, printRegressionMetrics


categoryList = {"Cold":0,
                "Fog":1,
                "Hail":2,
                "Precipitation":3,
                "Rain":4,
                "Snow":5,
                "Storm":6}

severityList = {"Light":7,
                "Moderate":8,
                "Heavy":9,
                "Severe":10,
                "Other":11,
                "UNK":12}

def plotWeatherType(rdd):

    rdd.map(lambda x: categoryList[x[2]]).collect

    output_x = np.array([])
    output_y = np.array([])

def getDistinctEnumTypes(dataRDD):

    print('Collecting all enumerated types...\n')
    print(f'All weather types: {dataRDD.map(lambda x: x[2]).distinct().collect()}\n')
    print(f'All severity types: {dataRDD.map(lambda x: x[3]).distinct().collect()}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Model Validation for Airplane Departure Prediction',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_set', type=str,                                         help="Path to the data set")
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Model Validation Airplane Departure Prediction')
    sc.setLogLevel('warn')

    rdd = readRDD(sc, 'data/processed/filtered_data_set')

    data = rdd.collect()





