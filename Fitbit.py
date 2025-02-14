import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm

df = pd.read_csv('daily_acivity.csv')

# Convert date columns to datetime
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])

# Extract the day of the week from the ActivityDate column
df['DayOfWeek'] = df['ActivityDate'].dt.day_name()

# Count the frequency of workouts for each day of the week
day_of_week_counts = df['DayOfWeek'].value_counts().sort_index()

# Plot the frequency of workouts for each day of the week
plt.figure(figsize=(10, 5))
day_of_week_counts.plot(kind='bar')
plt.xlabel('Day of the Week')
plt.ylabel('Frequency of Workouts')
plt.title('Frequency of Workouts for Each Day of the Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

unique_users = df['Id'].nunique()
print(f"Number of unique users: {unique_users}")

total_distance_per_user = df.groupby('Id')['TotalDistance'].sum()

total_distance_per_user.plot(kind='bar')
plt.xlabel('User ID')
plt.ylabel('Total Distance')
plt.title('Total Distance Registered by Fitbits for Each User')
plt.show()

def plot_calories_burnt(user_id):
    # Filter the DataFrame for the given user
    user_data = df[df['Id'] == user_id]
    
    # Determine the start and end dates
    start_date = user_data['ActivityDate'].min()
    end_date = user_data['ActivityDate'].max()
    
    # Plot the calories burnt on each day
    plt.figure(figsize=(10, 5))
    plt.plot(user_data['ActivityDate'], user_data['Calories'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Calories Burnt')
    plt.title(f'Calories Burnt Each Day for User {user_id}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

while True:
        user_id = int(input("\nEnter the user ID: "))
        if user_id in df['Id'].unique():
            plot_calories_burnt(user_id)
            break
        else:
            print("Invalid user ID. Please try again.")

# Prepare the data for the regression model
df['Id'] = df['Id'].astype('category')

# Define the dependent and independent variables
X = df[['TotalSteps', 'Id']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical variable into dummy/indicator variables
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['Calories']

# Ensure there are no missing values
X = X.dropna()
y = y[X.index]

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary of the regression results
print(model.summary())

def plot_regression_for_user(user_id):
    # Filter the DataFrame for the given user
    user_data = df[df['Id'] == user_id]
    
    # Define the dependent and independent variables
    X_user = user_data[['TotalSteps']]
    X_user = sm.add_constant(X_user)  # Adds a constant term to the predictor
    y_user = user_data['Calories']
    
    # Fit the model
    model_user = sm.OLS(y_user, X_user).fit()
    
    # Plot the scatterplot and regression line
    plt.figure(figsize=(10, 5))
    plt.scatter(user_data['TotalSteps'], user_data['Calories'], label='Data points')
    plt.plot(user_data['TotalSteps'], model_user.predict(X_user), color='red', label='Regression line')
    plt.xlabel('Total Steps')
    plt.ylabel('Calories Burnt')
    plt.title(f'Regression of Calories Burnt on Total Steps for User {user_id}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage of the function
plot_regression_for_user(1503960366)

# Additional creative insights and visualizations
# Example: Relationship between Total Distance and Calories Burnt
plt.figure(figsize=(10, 5))
plt.scatter(df['TotalDistance'], df['Calories'], alpha=0.5)
plt.xlabel('Total Distance')
plt.ylabel('Calories Burnt')
plt.title('Relationship between Total Distance and Calories Burnt')
plt.tight_layout()
plt.show()

# Example: Average Calories Burnt per Day of the Week
avg_calories_per_day = df.groupby('DayOfWeek')['Calories'].mean().sort_index()
plt.figure(figsize=(10, 5))
avg_calories_per_day.plot(kind='bar')
plt.xlabel('Day of the Week')
plt.ylabel('Average Calories Burnt')
plt.title('Average Calories Burnt per Day of the Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


