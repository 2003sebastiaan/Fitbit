import streamlit as st

st.set_page_config(
    page_title = "Fitbit Dashboard",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

##TIJDELIJKE ONZIN
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
st.bar_chart(class_counts.set_index("Class"))
