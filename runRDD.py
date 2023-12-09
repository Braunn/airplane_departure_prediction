from pyspark import SparkContext
from datetime import datetime
from os import path 

'''
TODO: fix corner case for mapping weather event with departure that happens across year boundry 
TODO: make enum types for indexing into weather and departure csvs for readability and debugging?
'''
def debugPrint(rdd, name):
    # Debug print out a few of them
    print(f"{name}.count() =",rdd.count(),"\n")
    print(f"{name}.takeSample(False, 5) =\n",rdd.takeSample(False, 5),"\n")

def swap(pair):
    return (pair[1], pair[0])

def weatherMakeACYearKeys(rdd):
    '''
    take in a weather rdd and 
    '''

def organizeDeparturesByACYear(rdd, N):
    '''
    takes a depature rdd of the form: [(index, (departure data N-tuple)),...]
    and combines keys so the returned RDD looks like:
    [((Origin, Year),[(datetime, index)]),...]

    this is used for mapping the weather events to departure data
    '''
    indexDatetime = 1
    indexOriginCode = 0

    # use this rdd to map departures to weather event. Only need time, space (origin airport code), 
    # and way back to element in data_set (index)
    # create keys e.g. [((Origin Airport Code, Year), (datetime, index)), ...]
    rdd = rdd.map(lambda x: ((x[1][indexOriginCode], x[1][indexDatetime].year),(x[1][indexDatetime], x[0])))

    # organize departure events based on (Origin Airport, Year)
    # [((Origin, Year),[(datetime, index)]),...]
    return rdd.aggregateByKey([],lambda alist, el: alist+[el], lambda alist, blist: alist+blist, numPartitions = N )


def organizeWeatherByACYear(rdd, N):
    '''
    takes a weather rdd of the form: [((Origin Code, Year),(weather data N-tuple)),...]
    and combines keys so the returned RDD looks like:
    [((Origin, Year),[(datetime event start, datetime event end, index)]),...]
    '''
    indexDatetimeStart = 3
    indexDatetimeEnd = 4
    indexOriginCode = 0

    # use this rdd to map departures to weather event. Only need time, space (origin airport code), 
    # and way back to element in data_set (index)
    # create keys e.g. [((Origin Airport Code, Year), (datetime start, datetime end, index)), ...]
    rdd = rdd.map(lambda x: ((x[1][indexOriginCode], x[1][indexDatetimeStart].year),(x[1][indexDatetimeStart], x[1][indexDatetimeEnd], x[0])))

    # organize departure events based on (Origin Airport, Year)
    # [((Origin, Year),[(datetime start, datetime end, index)]),...]
    return rdd.aggregateByKey([],lambda alist, el: alist+[el], lambda alist, blist: alist+blist, numPartitions = N )


def sortByTime(rdd, N):
    '''
    sort RDD by time
    expects the first element in RDD is a datetime object 
    N = number of partitions
    '''
    rdd.sort()

def joinDepartureWeather(weatherRDD, departureRDD):
    '''
    Sorts each RDD by time 
    joins by airport code
    uses binary search 
    '''
    pass


if __name__ == "__main__": 
    sc = SparkContext(appName='Airplane Departure Prediction')

    N = 30 # SET NUM PARTITIONS

    # Read in the airport departure statistics from CSV in to an RDD and split by commas
    departureRDD = sc.textFile("concatenated_data.csv").map(lambda x: x.split(","))
    
    # Strip the RDD down to the columns we care about: (origin airport, scheduled departure date, scheduled departure time, weather delay in minutes)
    # Andcombine the date and time in to one string
    departureRDD = departureRDD.map(lambda x: (x[17], x[1] + " " + x[5] + ":01", float(x[13])))

    # Convert that date/time string to a datetime object
    departureRDD = departureRDD.map(lambda x: (x[0], datetime.strptime(x[1], "%m/%d/%Y %H:%M:%S"), x[2]) )

    # Cache, this will be used as the basis for our actual data set 
    # [(index, (origin airport,  datetime, departure delay)), ]
    departureRDD = departureRDD.zipWithIndex().map(swap).cache()
    data_set = departureRDD

    # Debug print out a few of them
    print("departureRDD.count() =",departureRDD.count(),"\n")
    print("departureRDD.takeSample(False, 5) =\n",departureRDD.takeSample(False, 5),"\n")

    departureMapping = organizeDeparturesByACYear(departureRDD, N)
    debugPrint(departureMapping, 'departureMapping')


    # Read in the weather data from CSV in to an RDD and split by commas
    weatherRDD = sc.textFile("cleaned_weather_data.csv").map(lambda x: x.split(","))

    # Strip the RDD down to the columns we care about: (AirportCode,Type,Severity,StartTime(UTC),EndTime(UTC),Precipitation(in))
    weatherRDD = weatherRDD.map(lambda x: (x[7],x[1],x[2],x[3],x[4],float(x[5])))

    # Convert that date/time string to a datetime object
    weatherRDD = weatherRDD.map(lambda x: (x[0], x[1], x[2], datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), datetime.strptime(x[4], "%Y-%m-%d %H:%M:%S"), x[5]) )
    weatherRDD = weatherRDD.zipWithIndex().map(swap).cache()
    weatherMapping = organizeWeatherByACYear(weatherRDD, N)

    # Cache it
    weatherRDD = weatherRDD.cache()

    # Debug print out a few of them
    print("weatherRDD.count() =",weatherRDD.count(),"\n")
    print("weatherRDD.takeSample(False, 5) =\n",weatherRDD.takeSample(False, 5),"\n")

    # Join the two RDDs together?
    #departureWeatherRDD = departureRDD.join(weatherRDD)

    # Debug print out a few of them
    #print("departureWeatherRDD.count() =",departureWeatherRDD.count(),"\n")
    #print("departureWeatherRDD.takeSample(False, 5) =\n",departureWeatherRDD.takeSample(False, 5),"\n")
