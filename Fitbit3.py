import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
#import numpy as np


# Connect to the fitbit database
connection = sqlite3.connect("fitbit_database.db")
cursor = connection.cursor()

# Query to get the count of daily activities per user
query = """
SELECT Id, COUNT(*) as activity_count
FROM daily_activity
GROUP BY Id
"""
cursor.execute(query)

rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=['Id', 'activity_count'])

# Classify users based on activity count
def classify_user(activity_count):
    if activity_count <= 10:
        return 'Light user'
    elif 11 <= activity_count <= 15:
        return 'Moderate user'
    else:
        return 'Heavy user'

df['Class'] = df['activity_count'].apply(classify_user)

# Select only the Id and Class columns
result_df = df[['Id', 'Class']]


# Point 1: Compute the sleep duration for each moment of sleep
sleep_query = """
SELECT Id, logId, date as ActivityDate, SUM(value) as total_sleep_duration
FROM minute_sleep
GROUP BY Id, logId
"""
cursor.execute(sleep_query)
sleep_rows = cursor.fetchall()
sleep_df = pd.DataFrame(sleep_rows, columns=['Id', 'logId', 'ActivityDate', 'total_sleep_duration'])
sleep_df['ActivityDate'] = pd.to_datetime(sleep_df['ActivityDate']).dt.date
print(sleep_df[0:10])

# Point 2: perform a regression of active minutes against minutes of sleep
active_minutes_query = """
SELECT Id, ActivityDate, SUM(VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes) as TotalActiveMinutes
FROM daily_activity
GROUP BY Id, ActivityDate
"""
cursor.execute(active_minutes_query)
active_minutes_rows = cursor.fetchall()
active_minutes_df = pd.DataFrame(active_minutes_rows, columns=['Id', 'ActivityDate', 'total_active_minutes'])
active_minutes_df['ActivityDate'] = pd.to_datetime(active_minutes_df['ActivityDate']).dt.date
print(active_minutes_df.head())

merged_df = pd.merge(sleep_df, active_minutes_df, on=['Id'])
#merged_df = pd.merge(sleep_df, active_minutes_df, on=['Id', 'ActivityDate'])

X = merged_df['total_active_minutes'].values.reshape(-1, 1)
y = merged_df['total_sleep_duration'].values
model = LinearRegression()
model.fit(X, y)
print(f"Coefficient: {model.coef_}, Intercept: {model.intercept_}")

# Visualize the regression
# plt.scatter(X, y, color='blue')
# plt.plot(X, model.predict(X), color='red')
# plt.xlabel('Total Active Minutes')
# plt.ylabel('Total Sleep Duration')
# plt.title('Linear Regression: Active Minutes vs Sleep Duration')
# plt.show()

# Point 3: Use plotly to visualize the regression
sedentary_query = """
SELECT Id, ActivityDate, SedentaryMinutes
FROM daily_activity
"""
cursor.execute(sedentary_query)
# sedentary_rows = cursor.fetchall()
# sedentary_df = pd.DataFrame(sedentary_rows, columns=['Id', 'ActivityDate', 'SedentaryMinutes'])
# sedentary_df['ActivityDate'] = pd.to_datetime(sedentary_df['ActivityDate']).dt.date

# # Merge the two datasets on Id and ActivityDate
# merged_df = pd.merge(sleep_df, sedentary_df, on=['Id', 'ActivityDate'])
# print(merged_df)

# X = merged_df['SedentaryMinutes'].values.reshape(-1, 1)  # Independent variable
# y = merged_df['total_sleep_duration'].values  # Dependent variable

# # Fit the linear regression model
# model = LinearRegression()
# model.fit(X, y)

# # Get predictions
# y_pred = model.predict(X)

# # Print model parameters
# print(f"Coefficient: {model.coef_}, Model Intercept: {model.intercept_}")

residuals = y - model.predict(X)

# Histogram of residuals
# sns.histplot(residuals, kde=True)
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.title("Histogram of Residuals")
# plt.show()

# # QQ-Plot of residuals
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.title("QQ-Plot of Residuals")
# plt.show()

# Point 4: Divide the day into 4-hour blocks
def classify_time_of_day(hour):
    if 0 <= hour < 4:
        return "00-04"
    elif 4 <= hour < 8:
        return "04-08"
    elif 8 <= hour < 12:
        return "08-12"
    elif 12 <= hour < 16:
        return "12-16"
    elif 16 <= hour < 20:
        return "16-20"
    else:
        return "20-24"  
    
# Query to get hourly steps    
hourly_steps_query = """
SELECT Id, ActivityHour, StepTotal
FROM hourly_steps
"""
cursor.execute(hourly_steps_query)
hourly_steps_rows = cursor.fetchall()
hourly_steps_df = pd.DataFrame(hourly_steps_rows, columns=['Id', 'ActivityHour', 'StepTotal'])

# Convert ActivityHour to datetime and extract hour
hourly_steps_df['ActivityHour'] = pd.to_datetime(hourly_steps_df['ActivityHour'], format='%m/%d/%Y %I:%M:%S %p')
hourly_steps_df['Hour'] = hourly_steps_df['ActivityHour'].dt.hour

# Apply the function to classify time of day
hourly_steps_df['TimeBlock'] = hourly_steps_df['Hour'].apply(classify_time_of_day)

# Compute average steps for each time block
steps_summary = hourly_steps_df.groupby('TimeBlock')['StepTotal'].mean().reset_index()
steps_summary.rename(columns={'StepTotal': 'AverageSteps'}, inplace=True)
print(steps_summary)

# Query to get hourly calories
hourly_calories_query = """
SELECT Id, ActivityHour, Calories
FROM hourly_calories
"""
cursor.execute(hourly_calories_query)
hourly_calories_rows = cursor.fetchall()
hourly_calories_df = pd.DataFrame(hourly_calories_rows, columns=['Id', 'ActivityHour', 'Calories'])

hourly_calories_df['ActivityHour'] = pd.to_datetime(hourly_calories_df['ActivityHour'], format='%m/%d/%Y %I:%M:%S %p')
hourly_calories_df['Hour'] = hourly_calories_df['ActivityHour'].dt.hour

hourly_calories_df['TimeBlock'] = hourly_calories_df['Hour'].apply(classify_time_of_day)

calories_summary = hourly_calories_df.groupby('TimeBlock')['Calories'].mean().reset_index()
calories_summary.rename(columns={'Calories': 'AverageCalories'}, inplace=True)
print(calories_summary)

# Query to get minute sleep data
minute_sleep_query = """
SELECT Id, date as Timestamp, value as SleepMinutes
FROM minute_sleep
"""
cursor.execute(minute_sleep_query)
minute_sleep_rows = cursor.fetchall()
minute_sleep_df = pd.DataFrame(minute_sleep_rows, columns=['Id', 'Timestamp', 'SleepMinutes'])


minute_sleep_df['Timestamp'] = pd.to_datetime(minute_sleep_df['Timestamp'], format='%m/%d/%Y %I:%M:%S %p')
minute_sleep_df['Hour'] = minute_sleep_df['Timestamp'].dt.hour
minute_sleep_df['ActivityDate'] = minute_sleep_df['Timestamp'].dt.date

minute_sleep_df['TimeBlock'] = minute_sleep_df['Hour'].apply(classify_time_of_day)

individual_sleep = minute_sleep_df.groupby(['Id', 'ActivityDate', 'TimeBlock'])['SleepMinutes'].sum().reset_index()

sleep_summary = individual_sleep.groupby('TimeBlock')['SleepMinutes'].mean().reset_index()
sleep_summary.rename(columns={'SleepMinutes': 'AverageSleepMinutes'}, inplace=True)

print(sleep_summary)



# # Visualize the results in a barplot
# summary_df.plot(x='TimeBlock', y=['AverageSteps', 'AverageCalories', 'AverageSleepMinutes'], kind='bar')
# plt.xlabel('Time Block')
# plt.ylabel('Average')
# plt.title('Average Steps, Calories, and Sleep Minutes per 4-Hour Block')
# plt.show()