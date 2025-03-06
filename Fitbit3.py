import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np


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
print(sleep_df.head())

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

merged_df = pd.merge(sleep_df, active_minutes_df, on=['Id', 'ActivityDate'])
print(merged_df)

X = merged_df['total_active_minutes'].values.reshape(-1, 1)
y = merged_df['total_sleep_duration'].values
model = LinearRegression()
model.fit(X, y)
print(f"Coefficient: {model.coef_}, Intercept: {model.intercept_}")

# Visualize the regression
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Total Active Minutes')
plt.ylabel('Total Sleep Duration')
plt.title('Linear Regression: Active Minutes vs Sleep Duration')
plt.show()

# Point 3: Use plotly to visualize the regression
sedentary_query = """
SELECT Id, ActivityDate, SedentaryMinutes
FROM daily_activity
"""
cursor.execute(sedentary_query)
sedentary_rows = cursor.fetchall()
sedentary_df = pd.DataFrame(sedentary_rows, columns=['Id', 'ActivityDate', 'SedentaryMinutes'])
sedentary_df['ActivityDate'] = pd.to_datetime(sedentary_df['ActivityDate']).dt.date

# Merge the two datasets on Id and ActivityDate
merged_df = pd.merge(sleep_df, sedentary_df, on=['Id', 'ActivityDate'])
print(merged_df)
