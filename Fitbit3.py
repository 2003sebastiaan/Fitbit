import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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

print(result_df)

sleep_query = """
SELECT Id, logId, SUM(duration) as total_sleep_duration
FROM minute_sleep
GROUP BY Id, logId
"""
cursor.execute(sleep_query)
sleep_rows = cursor.fetchall()
sleep_df = pd.DataFrame(sleep_rows, columns=['Id', 'logId', 'total_sleep_duration'])
print(sleep_df)

active_minutes_query = """
SELECT Id, date, SUM(very_active_minutes + fairly_active_minutes + lightly_active_minutes) as total_active_minutes
FROM daily_activity
GROUP BY Id, date
"""
cursor.execute(active_minutes_query)
active_minutes_rows = cursor.fetchall()
active_minutes_df = pd.DataFrame(active_minutes_rows, columns=['Id', 'date', 'total_active_minutes'])

# Merge sleep duration and active minutes dataframes
merged_df = pd.merge(sleep_df, active_minutes_df, on=['Id', 'date'])
print(merged_df)