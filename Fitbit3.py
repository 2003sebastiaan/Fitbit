import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
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
#print(sleep_df[0:10])




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
#print(active_minutes_df.head())

merged_df = pd.merge(sleep_df, active_minutes_df, on=['Id'])
#merged_df = pd.merge(sleep_df, active_minutes_df, on=['Id', 'ActivityDate'])

X = merged_df['total_active_minutes'].values.reshape(-1, 1)
y = merged_df['total_sleep_duration'].values
model = LinearRegression()
model.fit(X, y)
#print(f"Coefficient: {model.coef_}, Intercept: {model.intercept_}")

#Visualize the regression
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
sedentary_rows = cursor.fetchall()
sedentary_df = pd.DataFrame(sedentary_rows, columns=['Id', 'ActivityDate', 'SedentaryMinutes'])
sedentary_df['ActivityDate'] = pd.to_datetime(sedentary_df['ActivityDate']).dt.date

# Merge the two datasets on Id and ActivityDate
merged_df = pd.merge(sleep_df, sedentary_df, on=['Id', 'ActivityDate'])
#print(merged_df)

X = merged_df['SedentaryMinutes'].values.reshape(-1, 1)  # Independent variable
y = merged_df['total_sleep_duration'].values  # Dependent variable

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)

# Print model parameters
#print(f"Coefficient: {model.coef_}, Model Intercept: {model.intercept_}")

residuals = y - model.predict(X)

#Histogram of residuals
# sns.histplot(residuals, kde=True)
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.title("Histogram of Residuals")
# plt.show()

# QQ-Plot of residuals
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
#print(steps_summary)

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
#print(calories_summary)

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

summary_df = pd.merge(steps_summary, calories_summary, on='TimeBlock')
summary_df = pd.merge(summary_df, sleep_summary, on='TimeBlock')

# Plot average steps per 4-hour block
# plt.figure(figsize=(10, 6))
# plt.bar(summary_df['TimeBlock'], summary_df['AverageSteps'], color='blue')
# plt.xlabel('Time Block')
# plt.ylabel('Average Steps')
# plt.title('Average Steps per 4-Hour Block')
# plt.show()

# # Plot average calories burnt per 4-hour block
# plt.figure(figsize=(10, 6))
# plt.bar(summary_df['TimeBlock'], summary_df['AverageCalories'], color='green')
# plt.xlabel('Time Block')
# plt.ylabel('Average Calories Burnt')
# plt.title('Average Calories Burnt per 4-Hour Block')
# plt.show()

# # Plot average sleep minutes per 4-hour block
# plt.figure(figsize=(10, 6))
# plt.bar(summary_df['TimeBlock'], summary_df['AverageSleepMinutes'], color='red')
# plt.xlabel('Time Block')
# plt.ylabel('Average Sleep Minutes')
# plt.title('Average Sleep Minutes per 4-Hour Block')
# plt.show()





# Point 5: heart rate and intensity data

def plot_heartrate_intensity(user_id):
    heart_rate_query = """
    SELECT Id, Time as ActivityHour, Value as HeartRate
    FROM heart_rate
    WHERE Id = ?
    """
    cursor.execute(heart_rate_query, (user_id,))
    heart_rate_rows = cursor.fetchall()
    heart_rate_df = pd.DataFrame(heart_rate_rows, columns=['Id', 'ActivityHour', 'HeartRate'])
    heart_rate_df['ActivityHour'] = pd.to_datetime(heart_rate_df['ActivityHour'], format='%m/%d/%Y %I:%M:%S %p')

    intensity_query = """
    SELECT Id, ActivityHour, TotalIntensity
    FROM hourly_intensity
    WHERE Id = ?
    """
    cursor.execute(intensity_query, (user_id,))
    intensity_rows = cursor.fetchall()
    intensity_df = pd.DataFrame(intensity_rows, columns=['Id', 'ActivityHour', 'TotalIntensity'])
    intensity_df['ActivityHour'] = pd.to_datetime(intensity_df['ActivityHour'], format='%m/%d/%Y %I:%M:%S %p')

    print(user_id)
    #erged_df = pd.merge(heart_rate_df, intensity_df, on=['Id', 'ActivityHour'])

      
    
# heart_rate_query = """
# SELECT * 
# FROM heart_rate
# """
# cursor.execute("PRAGMA table_info(heart_rate)")
# heart_rate_info = cursor.fetchall()
# print("heart_rate table structure:")
# for column in heart_rate_info:
#     print(column)

# intensity_query = """
# SELECT *
# FROM hourly_intensity
# """
# cursor.execute("PRAGMA table_info(hourly_intensity)")
# hourly_intensity_info = cursor.fetchall()
# print("hourly_intensity table structure:")
# for column in hourly_intensity_info:
#     print(column)

# Find a valid user_id that exists in both tables
cursor.execute("SELECT DISTINCT Id FROM heart_rate INTERSECT SELECT DISTINCT Id FROM hourly_intensity")
valid_ids = cursor.fetchall()

# Now call the function
user_id = 2022484408
#plot_heartrate_intensity(valid_ids[0][0])
#plot_heartrate_intensity(user_id)



#point 6: 
weather_data = pd.read_csv("Chicago_weather.csv", header=0)
print(weather_data.columns)

def dailysteps_precipitation():
    columns_precip = ["datetime", "precip"]
    weather_precip = weather_data[columns_precip].copy()
    weather_precip['datetime'] = pd.to_datetime(weather_precip['datetime']).dt.date

    daily_steps_query = """
    SELECT Id, ActivityHour, StepTotal
    FROM hourly_steps
    """
    cursor.execute(daily_steps_query)
    daily_steps_rows = cursor.fetchall()
    daily_steps_df = pd.DataFrame(daily_steps_rows, columns=['Id', 'ActivityHour', 'StepTotal'])
    print(daily_steps_df.head(15))

    # convert to total daily steps instead of hourly
    daily_steps_df["ActivityHour"] = pd.to_datetime(daily_steps_df["ActivityHour"], format='%m/%d/%Y %I:%M:%S %p')
    daily_steps_df["datetime"] = daily_steps_df["ActivityHour"].dt.date
    daily_steps_summary = daily_steps_df.groupby("datetime", as_index=False)["StepTotal"].mean()
       
    merged_df = pd.merge(daily_steps_summary, weather_precip, on="datetime", how="left")   
    return merged_df
    
def plot_steps_vs_precip(merged_df):
    fig = px.scatter(
        merged_df, 
        x="precip", 
        y="StepTotal",
        title="Correlation between Precipitation <br>and Daily Steps",
        labels={"precip": "Precipitation (mm)", "StepTotal": "Total Daily Steps"},
        trendline="ols"
    )
    fig.update_layout(
        xaxis=dict(title="Precipitation (mm)"),
        yaxis=dict(title="Total Daily Steps"),
        showlegend=False
    )
    fig.update_traces(
        marker=dict(color="#00B3BD"),
        line=dict(color="#005B8D")
    )
    fig.update_traces(
        hovertemplate="<b>Precipitation:</b> %{x}<br><b>Total Steps:</b> %{y:.0f}<extra></extra>"
    )
    fig.show()
   
#steps_vs_precip = dailysteps_precipitation()  # Ensure this function returns the DataFrame
#plot_steps_vs_precip(steps_vs_precip)  


def dailysteps_feelslike():
    columns_temp = ["datetime", "feelslike"]
    weather_temp = weather_data[columns_temp].copy()
    weather_temp["datetime"] = pd.to_datetime(weather_temp["datetime"]).dt.date
    
    daily_steps_query = """
    SELECT Id, ActivityHour, StepTotal
    FROM hourly_steps
    """
    cursor.execute(daily_steps_query)
    daily_steps_rows = cursor.fetchall()
    daily_steps_df = pd.DataFrame(daily_steps_rows, columns=['Id', 'ActivityHour', 'StepTotal'])

    # convert to total daily steps instead of hourly
    daily_steps_df["ActivityHour"] = pd.to_datetime(daily_steps_df["ActivityHour"], format='%m/%d/%Y %I:%M:%S %p')
    daily_steps_df["datetime"] = daily_steps_df["ActivityHour"].dt.date
    daily_steps_summary = daily_steps_df.groupby("datetime", as_index=False)["StepTotal"].sum()
       
    merged_df = pd.merge(daily_steps_summary, weather_temp, on="datetime", how="left")   
    return merged_df
    
def plot_steps_vs_temp(merged_df):
    fig = px.scatter(
        merged_df, 
        x="feelslike", 
        y="StepTotal",
        title="Correlation between Feels-Like Temperature <br>and Daily Steps",
        labels={"feelslike": "Feels-Like Temperature (°C)", "StepTotal": "Total Daily Steps"},
        trendline="ols"
    )
    fig.update_layout(
        xaxis=dict(title="Feels-Like Temperature (°C)"),
        yaxis=dict(title="Total Daily Steps"),
        showlegend=False
    )
    fig.update_traces(
        marker=dict(color="#FF5733"),
        line=dict(color="#8B0000")
    )
    fig.update_traces(
        hovertemplate="<b>Feels-Like Temp:</b> %{x}°C<br><b>Total Steps:</b> %{y:.0f}<extra></extra>"
    )
    fig.show()

#steps_vs_feelslike = dailysteps_feelslike()  # Ensure this function returns the DataFrame
#plot_steps_vs_temp(steps_vs_feelslike)


def sedentaryminutes_windspeed():
    columns_weather = ["datetime", "windspeed"]
    weather_data_wind = weather_data[columns_weather].copy()
    weather_data_wind['datetime'] = pd.to_datetime(weather_data_wind['datetime']).dt.date 
    print(weather_data_wind.head(15))
    
    daily_activity_query = """
    SELECT Id, ActivityDate as datetime, SedentaryMinutes
    FROM daily_activity
    """
    
    cursor.execute(daily_activity_query)
    daily_activity_rows = cursor.fetchall()
    daily_activity_df = pd.DataFrame(daily_activity_rows, columns=['Id', 'ActivityDate', 'SedentaryMinutes'])
    print(daily_activity_df.head(15))
    
    daily_activity_df["ActivityDate"] = pd.to_datetime(daily_activity_df["ActivityDate"]).dt.date
       
    merged_df = pd.merge(daily_activity_df, weather_data_wind, left_on="ActivityDate", right_on="datetime", how="left")
    print(merged_df.head(50))

    return merged_df

def plot_sedentary_vs_windspeed(merged_df):
    fig = px.scatter(
        merged_df,
        x="windspeed", 
        y="SedentaryMinutes",
        title="Correlation between Windspeed and Sedentary Minutes",
        labels={"windspeed": "Windspeed (km/h)", "SedentaryMinutes": "Total Sedentary Minutes"},
        trendline="ols"  # Add a trendline (Ordinary Least Squares)
    )
    fig.update_layout(
        xaxis=dict(title="Windspeed (km/h)"),
        yaxis=dict(title="Total Sedentary Minutes"),
        showlegend=False  # Hide legend if not necessary
    )
    fig.update_traces(
        marker=dict(color="#FF5733"),  # Color of points
        line=dict(color="#8B0000")     # Color of trendline
    )
    fig.update_traces(
        hovertemplate="<b>Windspeed:</b> %{x} km/h<br><b>Sedentary Minutes:</b> %{y}<extra></extra>"
    )
    fig.show()

sedentaryminutes_vs_windspeed = sedentaryminutes_windspeed()
plot_sedentary_vs_windspeed(sedentaryminutes_vs_windspeed)


def sedentaryminutes_temperature():
    columns_weather = ["datetime", "temp"]  
    weather_data_temp = weather_data[columns_weather].copy()
    weather_data_temp['datetime'] = pd.to_datetime(weather_data_temp['datetime']).dt.date

    # Fetch SedentaryMinutes data from daily_activity
    daily_activity_query = """
    SELECT Id, ActivityDate, SedentaryMinutes
    FROM daily_activity
    """
    cursor.execute(daily_activity_query)
    daily_activity_rows = cursor.fetchall()
    daily_activity_df = pd.DataFrame(daily_activity_rows, columns=['Id', 'ActivityDate', 'SedentaryMinutes'])

    # Convert ActivityDate to datetime and group by date to get daily SedentaryMinutes
    daily_activity_df["ActivityDate"] = pd.to_datetime(daily_activity_df["ActivityDate"]).dt.date
    daily_sedentary = daily_activity_df.groupby("ActivityDate", as_index=False)["SedentaryMinutes"].sum()

    # Merge Sedentary Minutes with temperature data
    merged_df = pd.merge(daily_sedentary, weather_data_temp, left_on="ActivityDate", right_on="datetime", how="left")
    return merged_df

def plot_sedentary_vs_temp(merged_df):
    fig = px.scatter(
        merged_df,
        x="temp",  # Or "feelslike" if you're using that instead
        y="SedentaryMinutes",
        title="Correlation between Temperature and Sedentary Minutes",
        labels={"temp": "Temperature (°C)", "SedentaryMinutes": "Total Sedentary Minutes"},
        trendline="ols"  # Add a trendline (Ordinary Least Squares)
    )
    fig.update_layout(
        xaxis=dict(title="Temperature (°C)"),
        yaxis=dict(title="Total Sedentary Minutes"),
        showlegend=False  # Hide legend if not necessary
    )
    fig.update_traces(
        marker=dict(color="#FF5733"),  # Color of points
        line=dict(color="#8B0000")     # Color of trendline
    )
    fig.update_traces(
        hovertemplate="<b>Temperature:</b> %{x}°C<br><b>Sedentary Minutes:</b> %{y}<extra></extra>"
    )
    fig.show()

#sedentaryminutes_vs_temp = sedentaryminutes_temperature()
#plot_sedentary_vs_temp(sedentaryminutes_vs_temp)

    
# temp vs total intensity
# feels like temp vs total active_minutes
# temp vs sedentary minutes
# precip vs sedentary minutes
# windspeed vs sedentary minutes
# sunrise - sunset vs sleep duration.    
    
