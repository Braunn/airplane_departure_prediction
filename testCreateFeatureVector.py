import random

from runRDD import createFeatureVector

categoryList = (("Cold",0),
                ("Fog",1),
                ("Hail",2),
                ("Precipitation",3),
                ("Rain",4),
                ("Snow",5),
                ("Storm",6))

severityList = (("Light",7),
                ("Moderate",8),
                ("Heavy",9),
                ("Severe",10),
                ("Other",11),
                ("UNK",12))

print("Running unit tests")

# Test every combination of categories and severities

for cat in categoryList:
    for sev in severityList:
        randomPrecipitation = random.uniform(0, 71.8)
        res = createFeatureVector(cat[0],sev[0],randomPrecipitation)

        for catIdx in range(0,7):
            if catIdx == cat[1]: # If this is the category we are testing, then make sure it is 1
                assert res[catIdx] == 1
            else: # If this isn't the category we are testing, then make sure it is set to 0
                assert res[catIdx] == 0

        for sevIdx in range(7,6):
            if sevIdx == sev[1]: # If this is the severity we are testing, then make sure it is 1
                assert res[sevIdx] == 1
            else: # If this isn't the severity we are testing, then make sure it is set to 0
                assert res[sevIdx] == 0

        assert res[13] == randomPrecipitation

        print("\t",cat[0],"/",sev[0],"/",randomPrecipitation,"= PASS")

# Test the degenerate case where it doesn't recognize either string

res = createFeatureVector("blah","blah",-5)

for catIdx in range(0,7):
    assert res[catIdx] == 0

for sevIdx in range(7,6):
    assert res[sevIdx] == 0

assert res[13] == -5

print("\t","blah","/","blah","/","-5","= PASS")
