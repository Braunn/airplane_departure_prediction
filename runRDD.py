from pyspark import SparkContext

sc = SparkContext(appName='Airplane Departure Prediction')
departureRDD = sc.textFile("concatenated_data.csv")

print("departureRDD.count() =",departureRDD.count())
print(departureRDD.take(5))

departureRDD = departureRDD.map(lambda x: x.split(","))

print("departureRDD.count() =",departureRDD.count())
print(departureRDD.take(5))
