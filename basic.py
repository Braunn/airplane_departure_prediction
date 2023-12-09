import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def filter_source_rows(df):
    return df[~df.apply(lambda row: row.astype(str).str.startswith(' SOURCE')).any(axis=1)]

result = pd.DataFrame()

# CONCATENATING THE AIRPORTS DATASET

# Put the path to the directory that contains the detailed departure statistics here
directory = "../"

print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"),"- Concatenating airport departure statistics")

for filename in os.scandir(directory):
    airportCode = filename.name[0:3]
    if filename.is_file():
        filePathStr = filename.path
        if "Detailed_Statistics_Departures" in filePathStr:
            print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"),"- Reading in",filePathStr)
            df = pd.read_csv(filePathStr,skiprows=7)
            # Apply the filter function to each DataFrame
            df = filter_source_rows(df)
            df['Origin Airport'] = airportCode
            # Concatenate along rows
            result = pd.concat([result, df], axis=0)

print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"),"- Writing concatenated airport departure statistics to file")

# Save the concatenated DataFrame to a new CSV file
result.to_csv('concatenated_data.csv', index=False, header=False)

#WORKING ON THE WEATHER DATA TO MERGE IT

print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"),"- Reading in weather events data")

# Load the data from the CSV file
file_path = directory+"WeatherEvents_Jan2016-Dec2022.csv" # Replace 'your_file_path.csv' with the actual path to your CSV file

df = pd.read_csv(file_path)

# Clean the "AirportCode" column by removing the "K"
df['AirportCode'] = df['AirportCode'].str.replace('K', '')

print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"),"- Writing weather data to file")

# Save the cleaned data to a new CSV file
output_file_path = 'cleaned_weather_data.csv'  # Replace 'cleaned_weather_data.csv' with the desired output file path
df.to_csv(output_file_path, index=False)

print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"),"- Done writing weather data to file")

#SEPARATE THE DATE AND TIME IN THE STARTTIME AND ENDTIME COLUMNS 
# Separate the date and time in the "StartTime(UTC)" column
df['StartDate'] = pd.to_datetime(df['StartTime(UTC)']).dt.date
df['StartTime'] = pd.to_datetime(df['StartTime(UTC)']).dt.time
# Separate the date and time in the "EndTime(UTC)" column
df['EndDate'] = pd.to_datetime(df['EndTime(UTC)']).dt.date
df['EndTime'] = pd.to_datetime(df['EndTime(UTC)']).dt.time
# Drop the original date-time columns
df = df.drop(['StartTime(UTC)', 'EndTime(UTC)'], axis=1)
# Reorder the columns for better readability
df = df[['EventId', 'Type', 'Severity', 'StartDate', 'StartTime', 'EndDate', 'EndTime',
         'Precipitation(in)', 'TimeZone', 'AirportCode', 'LocationLat', 'LocationLng',
         'City', 'County', 'State', 'ZipCode']]
# Save the modified data to a new CSV file
output_file_path = 'modified_data.csv'  # Replace 'modified_data.csv' with the desired output file path
df.to_csv(output_file_path, index=False)
# Display the modified data
# print("Modified Data:")
# print(df)

#MERGING THE MODIFIED DATA AND THE CONCATENATED DATA USING THE "AIRPORT CODE" AND "DESTINATION AIRPORT" COLUMNS IN THE RESPECTIVE DATASETS
# Read the datasets
df1 = pd.read_csv('modified_data.csv')
df2 = pd.read_csv('concatenated_data.csv')
# Merge the datasets on the common columns --> change the columns with which you want to merge the datasets 
# I wasn't able to figure out how to merge with 2 common columns. I used the airport codes. The date columns can also be used easily after separation. 
merged_df = pd.merge(df1, df2, left_on='AirportCode', right_on='Destination Airport', how='inner') 
# Optionally, you can drop the redundant columns (e.g., 'Destination Airport')
merged_df = merged_df.drop(['Destination Airport'], axis=1)
# Save the merged dataset to a new CSV file
merged_df.to_csv('merged_dataset.csv', index=False)
print(merged_df)

# Load the CSV file into a DataFrame
file_path = "/scratch/gandhi.ha/Project/merged_dataset.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to inspect the data
# print(df.head())

# Select relevant features and target
features = df[['Type', 'Severity', 'Scheduled elapsed time (Minutes)', 'Actual elapsed time (Minutes)']]
target = df['Departure delay (Minutes)']

# print(features)
# print(target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# print(X_train)
# Define preprocessor for categorical features
categorical_features = ['Type', 'Severity']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
print(categorical_transformer, 'categorical')


# Define preprocessor for numeric features
numeric_features = ['Scheduled elapsed time (Minutes)', 'Actual elapsed time (Minutes)']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
print(numeric_transformer,'numeric')

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
print(preprocessor, 'pre')
# Create a pipeline with the preprocessor and a linear regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Perform K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

# Print the error for each split
for i, mae in enumerate(-mae_scores):
    print(f'Mean Absolute Error (Split {i + 1}): {mae}')

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the departure delay on the test data
y_pred = model.predict(X_test)

# Evaluate the model on the test set
mae_test = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (Test Set): {mae_test}')






# Load new data for predictions
new_data = pd.read_csv("/scratch/gandhi.ha/Project/merged_dataset.csv")  # Replace with the actual file path for new data
# Assuming the new data has columns similar to the original data
# Select relevant features
new_features = new_data[['Type', 'Severity', 'Scheduled elapsed time (Minutes)', 'Actual elapsed time (Minutes)']]

# Take user input for Tail Number
user_tail_number = input("Enter Tail Number: ")

# Filter data for the provided Tail Number
user_data = new_data[new_data['Tail Number'] == user_tail_number]

if not user_data.empty:
    # Predict departure delay using the trained model
    predicted_delay = model.predict(new_features)

    # Display the results for the provided Tail Number
    print(f"Results for Tail Number: {user_tail_number}")
    for index, row in user_data.iterrows():
        print(f"  Flight Number: {row['Flight Number']}")
        print(f"  Departure Location: {row['City']}, {row['County']}, {row['State']}")
        print(f"  Arrival Location: {row['City']}, {row['County']}, {row['State']}")
        print(f"  Predicted Delay: {predicted_delay[index]:.2f} minutes")

else:
    print(f"No data found for Tail Number: {user_tail_number}")
#     # Display the results for the provided Tail Number
#     print(f"Results for Tail Number: {user_tail_number}")
#     for index, row in user_data.iterrows():
#         print(f"  Flight Number: {row['Flight Number']}, Predicted Delay: {predicted_delay[index]:.2f} minutes")

# else:
#     print(f"No data found for Tail Number: {user_tail_number}")






# # Select relevant features
# new_features = new_data[['Type', 'Severity', 'Scheduled elapsed time (Minutes)', 'Actual elapsed time (Minutes)']]

# # Predict departure delay using the trained model
# predicted_delay = model.predict(new_features)

# # Combine the results with the Carrier Code
# results = pd.DataFrame({
#     'Flight Number': new_data['Flight Number'],
#     'Predicted Delay (Minutes)': predicted_delay
# })

# # Print the results for each carrier
# unique_carriers = results['Flight Number'].unique()
# for carrier in unique_carriers:
#     carrier_results = results[results['Flight Number'] == carrier]
#     delayed_flights = carrier_results[carrier_results['Predicted Delay (Minutes)'] > 0]
    
#     if not delayed_flights.empty:
#         print(f"Carrier {carrier} has delayed flights:")
#         for index, row in delayed_flights.iterrows():
#             print(f"  Flight Number: {row['Flight Number']}, Predicted Delay: {row['Predicted Delay (Minutes)']:.2f} minutes")
#     else:
#         print(f"Carrier {carrier} has no delayed flights.")
