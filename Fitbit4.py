import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Connect to the fitbit database
connection = sqlite3.connect("fitbit_database.db")
cursor = connection.cursor()

# ===== DATABASE INSPECTION =====
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
table_names = [table[0] for table in tables]
print("Tables in the database:", table_names)

# ===== DATA COLLECTION =====
# Get core activity data first
daily_activity_df = pd.read_sql("SELECT * FROM daily_activity", connection)
daily_activity_df['ActivityDate'] = pd.to_datetime(daily_activity_df['ActivityDate'])


# ===== WEIGHT DATA PROCESSING =====
weight_query = "SELECT * FROM weight_log"
weight_log_df = pd.read_sql(weight_query, connection)
weight_log_df['Date'] = pd.to_datetime(weight_log_df['Date'])

# Rename 'Date' column to 'ActivityDate' for consistent merging
weight_log_df.rename(columns={'Date': 'ActivityDate'}, inplace=True)

# Handle missing WeightKg values
weight_log_df['WeightKg'] = weight_log_df.apply(
    lambda row: row['WeightKg'] if pd.notnull(row['WeightKg']) 
    else row['WeightPounds'] * 0.453592,
    axis=1
)

# Calculate missing Fat values
weight_log_df['NewFat'] = np.where(
    weight_log_df['Fat'].notna(),
    weight_log_df['Fat'],
    (1.20 * weight_log_df['BMI']) - 5.4
)

# ===== SLEEP DATA =====
sleep_query = """
SELECT 
    Id,
    DATE(date) AS ActivityDate,
    SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END) AS TotalMinutesAsleep,
    COUNT(*) AS TotalTimeInBed
FROM minute_sleep
GROUP BY Id, DATE(date)
"""
sleep_df = pd.read_sql(sleep_query, connection)
sleep_df['ActivityDate'] = pd.to_datetime(sleep_df['ActivityDate'])

# ===== HEART RATE DATA =====
resting_hr_query = """
SELECT 
    Id,
    DATE(Time) AS ActivityDate,
    MIN(Value) AS RestingHeartRate
FROM heart_rate
WHERE Value > 40  -- Remove unrealistic values
GROUP BY Id, DATE(Time)
"""
resting_hr_df = pd.read_sql(resting_hr_query, connection)
resting_hr_df['ActivityDate'] = pd.to_datetime(resting_hr_df['ActivityDate'])

# ===== DATA MERGING STRATEGY =====
# Start with daily activity as base
merged_df = daily_activity_df.copy()

# Merge other datasets with indicator flags
for df, name in [(sleep_df, 'sleep'), 
                (weight_log_df, 'weight'), 
                (resting_hr_df, 'heart_rate')]:
    merged_df = merged_df.merge(
        df,
        left_on=['Id', 'ActivityDate'],
        right_on=['Id', 'ActivityDate'],
        how='left',
        indicator=f'has_{name}'
    )

# ===== DATA CLEANING =====
# Convert all potential numeric columns
numeric_cols = ['TotalSteps', 'Calories', 'TotalMinutesAsleep', 
               'TotalTimeInBed', 'NewWeightKg', 'BMI', 'RestingHeartRate']
merged_df[numeric_cols] = merged_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Fill weight data forward within each user
merged_df['NewWeightKg'] = merged_df.groupby('Id')['NewWeightKg'].ffill()

# Calculate sleep efficiency safely
merged_df['TotalTimeInBed'] = merged_df['TotalTimeInBed'].clip(lower=1)
merged_df['SleepEfficiency'] = (
    merged_df['TotalMinutesAsleep'] / merged_df['TotalTimeInBed']
)

# Keep essential columns only
keep_columns = [
    'Id', 'ActivityDate', 'TotalSteps', 'Calories', 'VeryActiveMinutes',
    'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes',
    'TotalMinutesAsleep', 'TotalTimeInBed', 'SleepEfficiency',
    'NewWeightKg', 'BMI', 'RestingHeartRate'
]
merged_df = merged_df[keep_columns].rename(columns={'ActivityDate': 'Date'})

# Remove rows completely empty of health metrics
health_metrics = ['NewWeightKg', 'BMI', 'RestingHeartRate', 'TotalMinutesAsleep']
merged_df = merged_df.dropna(subset=health_metrics, how='all')

# ===== ANALYSIS & VISUALIZATION =====
if merged_df.empty:
    print("\nNo data available for analysis. Possible reasons:")
    print("- No overlapping dates between activity data and other metrics")
    print("- Missing core health metrics in all records")
    connection.close()
    exit()
else:
    print(f"\nAnalyzing {len(merged_df)} records with:")
    print(f"- {merged_df['has_weight'].mean():.1%} with weight data")
    print(f"- {merged_df['has_sleep'].mean():.1%} with sleep data")
    print(f"- {merged_df['has_heart_rate'].mean():.1%} with heart rate data")

# Statistical summary
stats = merged_df.groupby('Id').agg({
    'TotalSteps': ['mean', 'std'],
    'Calories': ['mean', 'max'],
    'TotalMinutesAsleep': ['mean', 'min', 'max'],
    'NewWeightKg': 'mean',
    'RestingHeartRate': 'mean',
    'BMI': 'mean'
}).reset_index()

print("\nAggregated Statistics:")
print(stats)

# Visualization setup
plt.figure(figsize=(20, 15))
plt.suptitle("Fitbit Data Analysis Dashboard", y=1.02, fontsize=18)

# Plot 1: Activity vs Weight
plt.subplot(2, 2, 1)
sns.scatterplot(
    data=merged_df,
    x='TotalSteps',
    y='NewWeightKg',
    hue='Id',
    size='Calories',
    palette='viridis',
    sizes=(20, 200),
    alpha=0.7
)
plt.title("Daily Steps vs Body Weight")

# Plot 2: Sleep Patterns
plt.subplot(2, 2, 2)
sns.histplot(
    data=merged_df,
    x='TotalMinutesAsleep',
    hue='Id',
    element='step',
    palette='coolwarm',
    bins=20
)
plt.title("Sleep Duration Distribution")

# Plot 3: Activity Composition
plt.subplot(2, 2, 3)
activity_cols = ['VeryActiveMinutes', 'FairlyActiveMinutes',
                'LightlyActiveMinutes', 'SedentaryMinutes']
melted = merged_df.melt(
    id_vars=['Id', 'Date'],
    value_vars=activity_cols,
    var_name='Activity',
    value_name='Minutes'
)
sns.lineplot(
    x='Date',
    y='Minutes',
    hue='Activity',
    data=melted,
    palette='Set2',
    estimator='mean'
)
plt.title("Activity Trends Over Time")

# Plot 4: Health Correlations
plt.subplot(2, 2, 4)
corr_matrix = merged_df[['NewWeightKg', 'BMI', 'RestingHeartRate',
                        'TotalMinutesAsleep', 'TotalSteps']].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='icefire',
    vmin=-1,
    vmax=1,
    linewidths=.5
)
plt.title("Health Metrics Correlation Matrix")

plt.tight_layout()
plt.show()

connection.close()
print("Analysis complete! Database connection closed.")