import streamlit as st

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

# Date Filtering
st.sidebar.header("Date Range")
min_date = pd.to_datetime('2023-01-01').date()  # Update with your actual min date
max_date = pd.to_datetime('2023-12-31').date()  # Update with your actual max date
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
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


