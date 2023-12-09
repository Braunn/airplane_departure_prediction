from pyspark import SparkContext
from datetime import datetime

sc = SparkContext(appName='Airplane Departure Prediction')

# Read in the airport departure statistics from CSV in to an RDD and split by commas
departureRDD = sc.textFile("concatenated_data.csv").map(lambda x: x.split(","))

# Strip the RDD down to the columns we care about: (origin airport, (scheduled departure date, scheduled departure time, weather delay in minutes))
departureRDD = departureRDD.map(lambda x: (x[17], (x[1], x[5], float(x[13]))) )

# Combine the date and time in to one string
departureRDD = departureRDD.mapValues(lambda x: (x[0] + " " + x[1] + ":01", x[2]) )

# Convert that date/time string to a datetime object
departureRDD = departureRDD.mapValues(lambda x: (datetime.strptime(x[0], "%m/%d/%Y %H:%M:%S"), x[1]) )

# Cache it
departureRDD = departureRDD.cache()

# Debug print out a few of them
print("departureRDD.count() =",departureRDD.count(),"\n")
print("departureRDD.takeSample(False, 5) =\n",departureRDD.takeSample(False, 5),"\n")

# Read in the weather data from CSV in to an RDD and split by commas
weatherRDD = sc.textFile("cleaned_weather_data.csv").map(lambda x: x.split(","))

# Strip the RDD down to the columns we care about: (AirportCode,(Type,Severity,StartTime(UTC),EndTime(UTC),Precipitation(in)))
weatherRDD = weatherRDD.map(lambda x: (x[7],(x[1],x[2],x[3],x[4],float(x[5]))))

# Convert that date/time string to a datetime object
weatherRDD = weatherRDD.mapValues(lambda x: (x[0], x[1], datetime.strptime(x[2], "%Y-%m-%d %H:%M:%S"), datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), x[4]) )

# Cache it
weatherRDD = weatherRDD.cache()

# Debug print out a few of them
print("weatherRDD.count() =",weatherRDD.count(),"\n")
print("weatherRDD.takeSample(False, 5) =\n",weatherRDD.takeSample(False, 5),"\n")

# Join the two RDDs together?
departureWeatherRDD = departureRDD.join(weatherRDD)

# Debug print out a few of them
print("departureWeatherRDD.count() =",departureWeatherRDD.count(),"\n")
print("departureWeatherRDD.takeSample(False, 5) =\n",departureWeatherRDD.takeSample(False, 5),"\n")
