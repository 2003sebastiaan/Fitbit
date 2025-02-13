import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm   

df = pd.read_csv('daily_acivity.csv')

unique_users = df['Id'].nunique()
print(f"Number of unique users: {unique_users}")

total_distance_per_user = df.groupby('Id')['TotalDistance'].sum()

total_distance_per_user.plot(kind='bar')
# plt.xlabel('User ID')
# plt.xticks(rotation=20)
# plt.ylabel('Total Distance')
# plt.title('Total Distance Registered by Fitbits for Each User')
# #plt.show()

def plot_calories_burnt(user_id):
    # Convert date columns to datetime
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    
    # Filter the DataFrame for the given user
    user_data = df[df['Id'] == user_id]
    
    # Determine the start and end dates
    start_date = user_data['ActivityDate'].min()
    end_date = user_data['ActivityDate'].max()
    
    # Plot the calories burnt on each day
    # plt.figure(figsize=(10, 5))
    # plt.plot(user_data['ActivityDate'], user_data['Calories'], marker='o')
    # plt.xlabel('Date')
    # plt.ylabel('Calories Burnt')
    # plt.title(f'Calories Burnt Each Day for User {user_id}')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # #plt.show()

while True:
    user_id = int(input("\nEnter the user ID: "))
    if user_id in df['Id'].values:
        plot_calories_burnt(user_id)
        break
    else:
        print("Invalid user ID. Please try again.")

# Add a column for the day of the week
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
df['DayOfWeek'] = df['ActivityDate'].dt.day_name()

# Count the frequency of workouts on each day of the week
day_of_week_counts = df['DayOfWeek'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Plot the frequency of workouts on each day of the week
plt.figure(figsize=(10, 5))
day_of_week_counts.plot(kind='bar')
plt.xlabel('Day of the Week')
plt.ylabel('Frequency of Workouts')
plt.title('Frequency of Workouts on Each Day of the Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

