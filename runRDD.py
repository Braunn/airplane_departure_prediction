from pyspark import SparkContext

sc = SparkContext(appName='Airplane Departure Prediction')
departureRDD = sc.textFile("concatenated_data.csv")

print("departureRDD.count() =",departureRDD.count())
print(departureRDD.take(5))

departureRDD = departureRDD.map(lambda x: x.split(","))

print("departureRDD.count() =",departureRDD.count())
print(departureRDD.take(5))

departureRDD = departureRDD.map(lambda x: (x[4], x[1], x[5], x[13]) )

print("departureRDD.count() =",departureRDD.count())
print(departureRDD.take(5))
