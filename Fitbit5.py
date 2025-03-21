import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Fitbit Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Connect to the fitbit database
connection = sqlite3.connect("fitbit_database.db")
cursor = connection.cursor()

# ===== NEW DATA COLLECTION SECTION ===== (MUST COME FIRST!)
# Sleep data
sleep_query = """
SELECT 
    Id,
    DATE(date) AS Date,
    SUM(CASE WHEN value = 1 THEN 1 ELSE 0 END) AS TotalMinutesAsleep,
    COUNT(*) AS TotalTimeInBed
FROM minute_sleep
GROUP BY Id, DATE(date)
"""
sleep_df = pd.read_sql(sleep_query, connection)
sleep_df['Date'] = pd.to_datetime(sleep_df['Date'])

# Activity data
activity_df = pd.read_sql("SELECT * FROM daily_activity", connection)
activity_df['Date'] = pd.to_datetime(activity_df['ActivityDate'])
activity_df = activity_df.drop('ActivityDate', axis=1)

# Merge
merged_df = activity_df.merge(sleep_df, on=['Id', 'Date'], how='left')

# Ensure numeric types and handle missing values
merged_df['TotalTimeInBed'] = pd.to_numeric(merged_df['TotalTimeInBed'], errors='coerce').fillna(1)
merged_df['TotalMinutesAsleep'] = pd.to_numeric(merged_df['TotalMinutesAsleep'], errors='coerce')

# Then calculate sleep efficiency
merged_df['SleepEfficiency'] = (
    merged_df['TotalMinutesAsleep'] / 
    merged_df['TotalTimeInBed'].replace(0, np.nan)
).clip(0, 1)

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

# Count occurrences of each class
class_counts = df['Class'].value_counts().reset_index()
class_counts.columns = ['Class', 'Count']

# Plot bar chart with Streamlit
# st.bar_chart(class_counts.set_index("Class"))


# Sidebar Setup
st.sidebar.header("Participant Selection")
user_ids = df['Id'].unique().tolist()
selected_id = st.sidebar.selectbox("Select Participant ID", user_ids)
# Existing sidebar code...
st.sidebar.header("Analysis Parameters")
analysis_type = st.sidebar.selectbox(
    "Sleep Analysis Focus",
    options=["Individual Patterns", "Group Trends", "Correlation Explorer"]
)

# In your date range setup, remove .date() conversions
min_date = pd.to_datetime('2023-01-01')  # No .date()
max_date = pd.to_datetime('2023-12-31')  # No .date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date.date(), max_date.date()),  # Convert to date objects here
    min_value=min_date.date(),
    max_value=max_date.date()
)

# Main Content
st.header("Research Overview")

# General Stats Columns
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Participants", df['Id'].nunique())
with col2:
    st.metric("Average Activities", round(df['activity_count'].mean(), 1))
with col3:
    st.metric("Most Common Class", class_counts.iloc[0]['Class'])

# Existing classification chart
st.subheader("User Activity Classification")
st.bar_chart(class_counts.set_index("Class"))

# Individual Participant Analysis
st.header(f"Participant {selected_id} Analysis")

# Get individual data
individual_data = df[df['Id'] == selected_id]

# Display individual metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Activity Count", individual_data['activity_count'].values[0])
with col2:
    st.metric("Classification", individual_data['Class'].values[0])

# Individual Sleep Patterns
if analysis_type == "Individual Patterns":
    st.subheader(f"Sleep Patterns for Participant {selected_id}")
    
    user_sleep = merged_df[
    (merged_df['Id'] == selected_id) &
    (merged_df['Date'].between(
        pd.to_datetime(date_range[0]),  # Convert to datetime64
        pd.to_datetime(date_range[1])
    ))
]
    
    if not user_sleep.empty:
        fig = px.line(user_sleep, x='Date', y='TotalMinutesAsleep',
                     hover_data=['TotalSteps', 'Calories', 'TotalDistance'],
                     title="Sleep Duration Timeline")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sleep metrics columns
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_sleep = user_sleep['TotalMinutesAsleep'].mean()/60
            st.metric("Avg Sleep", f"{avg_sleep:.1f} hrs")
        with col2:
            efficiency = user_sleep['SleepEfficiency'].mean()*100
            st.metric("Efficiency", f"{efficiency:.1f}%")
        with col3:
            corr = user_sleep['TotalMinutesAsleep'].corr(user_sleep['SedentaryMinutes'])
            st.metric("Sedentary Correlation", f"{corr:.2f}")
    else:
        st.warning("No sleep data available")




