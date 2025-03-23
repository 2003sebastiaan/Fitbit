import streamlit as st
import numpy as np
import sqlite3
import pandas as pd
import plotly.express as px

# Must be first Streamlit command
st.set_page_config(
    page_title="Fitbit Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Connect to the fitbit database
connection = sqlite3.connect("fitbit_database.db")
cursor = connection.cursor()

# ===== DATA COLLECTION =====
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

# Merge data
merged_df = activity_df.merge(sleep_df, on=['Id', 'Date'], how='left')

# Data cleaning
merged_df['TotalTimeInBed'] = pd.to_numeric(merged_df['TotalTimeInBed'], errors='coerce').fillna(1)
merged_df['TotalMinutesAsleep'] = pd.to_numeric(merged_df['TotalMinutesAsleep'], errors='coerce')
merged_df['SleepEfficiency'] = (merged_df['TotalMinutesAsleep'] / 
                               merged_df['TotalTimeInBed'].replace(0, np.nan)).clip(0, 1)

# User classification
query = "SELECT Id, COUNT(*) as activity_count FROM daily_activity GROUP BY Id"
df = pd.read_sql(query, connection)
df['Class'] = df['activity_count'].apply(lambda x: 'Light user' if x <=10 else 'Moderate user' if x <=15 else 'Heavy user')
class_counts = df['Class'].value_counts().reset_index()
class_counts.columns = ['Class', 'Count']

# Add class information to merged data
merged_df = merged_df.merge(df[['Id', 'Class']], on='Id', how='left')

# ===== WEATHER ANALYSIS =====
weather_merged = pd.DataFrame()
try:
    weather_df = pd.read_csv("Chicago_weather.csv")
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    
    # Prepare activity data with proper datetime type
    daily_steps = merged_df.groupby('Date')[['TotalSteps', 'SedentaryMinutes']].mean().reset_index()
    daily_steps['Date'] = pd.to_datetime(daily_steps['Date'])
    
    # Merge with weather data
    weather_merged = daily_steps.merge(
        weather_df,
        left_on='Date',
        right_on='datetime',
        how='left'
    )
except Exception as e:
    st.warning(f"Could not load weather data: {str(e)}")

# ===== SIDEBAR =====
st.sidebar.header("Participant Selection")
user_ids = df['Id'].unique().tolist()
selected_id = st.sidebar.selectbox("Select Participant ID", user_ids)

st.sidebar.header("Date Range")
min_date = pd.to_datetime('2016-01-01')
max_date = pd.to_datetime('2023-12-31')
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

# ===== MAIN CONTENT =====
tab1, tab2, tab3 = st.tabs(["General Overview", "User Groups Analysis", "Individual Analysis"])

with tab1:
    # General Overview Tab
    st.header("Research Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Participants", df['Id'].nunique())
    with col2:
        st.metric("Average Activities", round(df['activity_count'].mean(), 1))
    with col3:
        st.metric("Most Common Class", class_counts.iloc[0]['Class'])
    
    st.subheader("User Activity Classification")
    
    class_colors = {
        'Light user': '#1f77b4',    # Blue
        'Moderate user': '#2ca02c', # Green
        'Heavy user': '#d62728'     # Red
    }
    
    fig = px.bar(
        class_counts,
        x='Class',
        y='Count',
        color='Class',
        color_discrete_map=class_colors,
        title="User Classification Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("ℹ️ Classification Criteria"):
        st.markdown("""
        **Users are categorized based on their activity tracking consistency:**
        - <span style="color: #1f77b4">● Light User: 1-10 activity days</span>
        - <span style="color: #2ca02c">● Moderate User: 11-15 activity days</span>
        - <span style="color: #d62728">● Heavy User: 16+ activity days</span>
        """, unsafe_allow_html=True)
        
        threshold_df = pd.DataFrame({
            'User Class': ['Light', 'Moderate', 'Heavy'],
            'Minimum Days': [1, 11, 16],
            'Maximum Days': [10, 15, 'No limit']
        })
        st.dataframe(threshold_df, hide_index=True, use_container_width=True)

    # Improved Activity Trends Plot
    st.subheader("Activity Composition Over Time")
    
    # Define activity colors matching class scheme
    activity_colors = {
        'VeryActiveMinutes': '#d62728',    # Red (Heavy user)
        'FairlyActiveMinutes': '#2ca02c',  # Green (Moderate user)
        'LightlyActiveMinutes': '#1f77b4', # Blue (Light user)
        'SedentaryMinutes': '#7f7f7f'      # Grey
    }
    
    # Create weekly averages
    weekly_avg = merged_df.resample('W', on='Date').agg({
        'VeryActiveMinutes': 'mean',
        'FairlyActiveMinutes': 'mean',
        'LightlyActiveMinutes': 'mean',
        'SedentaryMinutes': 'mean'
    }).reset_index()
    
    # Melt for plotting
    melted = weekly_avg.melt(
        id_vars=['Date'],
        value_vars=activity_colors.keys(),
        var_name='Activity',
        value_name='Minutes'
    )
    
    # Create cleaner labels
    activity_labels = {
        'VeryActiveMinutes': 'Very Active',
        'FairlyActiveMinutes': 'Fairly Active',
        'LightlyActiveMinutes': 'Lightly Active',
        'SedentaryMinutes': 'Sedentary'
    }
    melted['Activity'] = melted['Activity'].map(activity_labels)
    
    fig = px.line(melted,
                 x='Date',
                 y='Minutes',
                 color='Activity',
                 color_discrete_map=activity_colors,
                 labels={'Minutes': 'Average Weekly Minutes'},
                 title="Weekly Activity Trends")
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Activity Minutes',
        hovermode='x unified',
        legend_title='Activity Type',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
    if not weather_merged.empty:
        st.subheader("Weather Impact Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(weather_merged, x='temp', y='TotalSteps',
                            trendline="ols", title="Temperature vs Daily Steps")
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.scatter(weather_merged, x='precip', y='SedentaryMinutes',
                            trendline="ols", title="Precipitation vs Sedentary Time")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.scatter(weather_merged, x='feelslike', y='TotalSteps',
                            trendline="ols", title="Feels-Like Temperature vs Steps")
            st.plotly_chart(fig, use_container_width=True)
            
            fig = px.scatter(weather_merged, x='windspeed', y='SedentaryMinutes',
                            trendline="ols", title="Wind Speed vs Sedentary Time")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    # User Groups Analysis Tab
    st.header("User Group Trends")
    
    grouped = merged_df.groupby('Class').agg({
        'TotalSteps': 'mean',
        'Calories': 'mean',
        'TotalMinutesAsleep': 'mean',
        'SleepEfficiency': 'mean'
    }).reset_index()
    
    st.subheader("Average Metrics by User Class")
    st.dataframe(grouped.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    metrics = ['TotalSteps', 'Calories', 'TotalMinutesAsleep', 'SleepEfficiency']
    for metric in metrics:
        fig = px.bar(grouped, x='Class', y=metric, 
                    title=f"Average {metric} by User Class", color='Class')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Temporal Trends by Class")
    grouped_time = merged_df.groupby(['Date', 'Class']).mean(numeric_only=True).reset_index()
    
    fig = px.line(grouped_time, x='Date', y='TotalSteps', color='Class',
                 title="Daily Step Trends by User Class")
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.line(grouped_time, x='Date', y='TotalMinutesAsleep', color='Class',
                 title="Sleep Duration Trends by User Class")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Individual Analysis Tab
    st.header(f"Participant Analysis: {selected_id}")
    
    individual_data = df[df['Id'] == selected_id]
    user_activity = merged_df[merged_df['Id'] == selected_id]
    user_sleep = user_activity[(user_activity['Date'] >= pd.to_datetime(date_range[0])) & 
                              (user_activity['Date'] <= pd.to_datetime(date_range[1]))]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Activity Count", individual_data['activity_count'].values[0])
    with col2:
        st.metric("User Classification", individual_data['Class'].values[0])
    
    st.subheader("Activity & Sleep Timeline")
    if not user_sleep.empty:
        fig = px.line(user_sleep, x='Date', y='TotalSteps', 
                     title="Daily Step Count", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.line(user_sleep, x='Date', y='TotalMinutesAsleep', 
                     title="Sleep Duration", markers=True,
                     hover_data=['TotalTimeInBed', 'SleepEfficiency'])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Sleep Efficiency Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            eff = user_sleep['SleepEfficiency'].mean()*100
            st.metric("Avg Sleep Efficiency", f"{eff:.1f}%" if not pd.isna(eff) else "N/A")
        with col2:
            try:
                best_day = user_sleep.loc[user_sleep['SleepEfficiency'].idxmax()]['Date'].strftime('%b %d')
                st.metric("Best Sleep Day", best_day)
            except ValueError:
                st.metric("Best Sleep Day", "N/A")
        with col3:
            corr = user_sleep['TotalMinutesAsleep'].corr(user_sleep['TotalSteps'])
            st.metric("Correlation with Steps", f"{corr:.2f}" if not pd.isna(corr) else "N/A")
    else:
        st.warning("No sleep data available for selected date range")

# Close database connection
connection.close()
