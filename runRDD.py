from pyspark import SparkContext
from datetime import datetime, timedelta
from os import path
from bisect import bisect 
import pdb

'''
TODO: fix corner case for mapping weather event with departure that happens across year boundry 
TODO: make enum types for indexing into weather and departure csvs for readability and debugging?
TODO: what is causing "input split" and is it a problem?
TODO: look into data more to determine what most important features about a storm are
TODO: add features for storm near arrival airport
TODO: *histogram storm duration to justify storm window?
TODO: speedup storm selection by changing datetime hh:mm to float in hours or smth? im thinking the datetime object is clunky 
'''

def featureSelection(pair):
    '''
    This function is for the final step after merging the weather events into the departure RDD.
    After the join this RDD will have a tuple in it's elements of the form (departure data, weather data).
    The "weather data" element may have a None instead of weather data which is handled here.
    Also handles "feature selection" slicing out unused data 
    '''
    # expects departure tuple = (Origin Airport Code, datetime scheduled departure, weather delay)
    departureTuple = pair[0][:2] 
    if pair[1] is None:
        # use default values for weather:
        # (Type,Severity,Precipitation(in))
        return (pair[0][2], (departureTuple + ('None','None', 0)))
    else:
        weatherTuple = pair[1][1:3] + pair[1][5:6] # take slice of tuple to avoid concat issues b/w tuple and float which happens when you use pair[1][5]
        return (pair[0][2], (departureTuple + weatherTuple))


def flatMapHelper(x):
    if type(x) is list:
        return x
    else:
        return [x]

def testMapping(w,d):
    #w = [(datetime.now(),datetime.now(), 1), (datetime.now(),datetime.now()+timedelta(seconds=4), 2)]
    #d = [(datetime.now(),0)]

    mappings = matchWeatherToDeparture(w,d)
    print(mappings)

def matchWeatherToDeparture(weather, departure):
        '''
        This function matches the closest storm/element in the weather data to each 
        element of the departure data. Closest storm is determined by closest start 
        time of storm/event to departure time that is w/i a one hour window. 
        Input:
            weather  : N element list, an element = (start time, end time, j)
            depature : M element list, an element = (scheduled departure time, i)
        Output:
            mappings : M element list, an element k = (i,j), this maps 
            each element in the departure array to the weather array
        '''

        timeWindow = timedelta(hours = 1) 
        mappings = []

        # naive method (probably can do some insertion sort type thing for step 2): 
        # 1. sort the smaller array weather data O(N log N) 
        # 2. binary search larger array O(log M)
        weather.sort() # sorts by 1st el of tuple
        nWeather = len(weather)
        for d in departure:
            index = bisect(weather, (d[0],)) # get index of event immediately after scheduled departure
            
            times = [(timeWindow, -2)]
            if index != 0:
                # get the event immediately before the departure
                times = times + [(abs(d[0] - weather[index -1][0]), -1)]
            if index < nWeather:
                # get the event immediately after the departure 
                times = times + [(abs(d[0] - weather[index][0]),0)]
            
            times.sort()
            closestIndex = times[0][1]

            # if 1 hour is the closest skip this departure
            # otherwise get the indexes
            if closestIndex != -2:
                mappings.append((d[1], weather[index + closestIndex][2]))
        
        return mappings 

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
    #def main(sc):
    sc = SparkContext(appName='Airplane Departure Prediction')
    sc.setLogLevel("ERROR") # Im seeing a ton of info messages without this, I messed up my spark config somehow -NCA

    N = 5 # SET NUM PARTITIONS

    departure_file = "concatenated_data_short.csv" #"concatenated_data.csv"
    weather_file = "cleaned_weather_data_short.csv" #"cleaned_weather_data.csv"

    # mute garabage collection?
    '''
    spark.cleaner.referenceTracking false
    spark.cleaner.referenceTracking.blocking false
    spark.cleaner.referenceTracking.blocking.shuffle false
    spark.cleaner.referenceTracking.cleanCheckpoints false 
    '''

    # Read in the airport departure statistics from CSV in to an RDD and split by commas
    departureRDD = sc.textFile(departure_file).map(lambda x: x.split(","))

    # Strip the RDD down to the columns we care about: (origin airport, scheduled departure date, scheduled departure time, weather delay in minutes)
    # Andcombine the date and time in to one string
    departureRDD = departureRDD.map(lambda x: (x[17], x[1] + " " + x[5] + ":01", float(x[13])))

    # Convert that date/time string to a datetime object
    departureRDD = departureRDD.map(lambda x: (x[0], datetime.strptime(x[1], "%m/%d/%Y %H:%M:%S"), x[2]) )

    # Cache, this will be used as the basis for our actual data set 
    # [(index, (origin airport,  datetime, departure delay)), ]
    departureRDD = departureRDD.zipWithIndex().map(swap).cache()

    # Debug print out a few of them
    #print("departureRDD.count() =",departureRDD.count(),"\n")
    #print("departureRDD.takeSample(False, 5) =\n",departureRDD.takeSample(False, 5),"\n")

    departureMapping = organizeDeparturesByACYear(departureRDD, N)
    debugPrint(departureMapping, 'departureMapping')


    # Read in the weather data from CSV in to an RDD and split by commas
    weatherRDD = sc.textFile(weather_file).map(lambda x: x.split(","))

    # Strip the RDD down to the columns we care about: (AirportCode,Type,Severity,StartTime(UTC),EndTime(UTC),Precipitation(in))
    weatherRDD = weatherRDD.map(lambda x: (x[7],x[1],x[2],x[3],x[4],float(x[5])))

    # Convert that date/time string to a datetime object
    weatherRDD = weatherRDD.map(lambda x: (x[0], x[1], x[2], datetime.strptime(x[3], "%Y-%m-%d %H:%M:%S"), datetime.strptime(x[4], "%Y-%m-%d %H:%M:%S"), x[5]) )
    weatherRDD = weatherRDD.zipWithIndex().map(swap).cache()
    weatherMapping = organizeWeatherByACYear(weatherRDD, N)
    weatherMapping.cache()

    debugPrint(weatherRDD, 'weatherRDD')
    debugPrint(weatherMapping, 'weatherMapping')

    # create the mapping b/w departure and weather event
    mappingRDD = weatherMapping.join(departureMapping, numPartitions = N) # removes keys that are present in weatherMapping and absent in departure mapping 
    mappingRDD = mappingRDD.mapValues(lambda arrayPair: matchWeatherToDeparture(arrayPair[0], arrayPair[1]) )
    debugPrint(mappingRDD, 'mappingRDD')

    # Prep weather RDD for join with departure RDD
    # 1. add departure indexes to weather data, also remove keys in weather data 
    # that arent mapped to a departure. resulting rdd = [(weather index, (weather data, departure index(es))), ...]
    # 2. get rid of the weather indexes, we are all done with those. rdd = [(weather data, departure index(es)), ...]
    # 3. handle 1 weather event being mapped to 1 or multiple departure events. rdd = [(weather data, departure index), ...]
    # 4. swap pair so resulting rdd is [(departure index, weather data),...]
    mappingRDD = mappingRDD.values().flatMap(flatMapHelper).map(swap) # [(departure index, weather index), ...] -> [(weather index, departure index), ...]
    weatherRDD = weatherRDD.join(mappingRDD, numPartitions=N)\
                    .values()\
                    .flatMapValues(flatMapHelper)\
                    .map(swap)\
                    .partitionBy(N)
    

    # Join the two RDDs together
    data_set = departureRDD.leftOuterJoin(weatherRDD, numPartitions=N)\
                .mapValues(featureSelection)\
                .values()\
                .cache()

    # Debug print out a few of them
    debugPrint(data_set, 'data_set')
    #return data_set, weatherRDD