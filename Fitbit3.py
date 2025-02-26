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

# Connect to the covid database
connection = sqlite3.connect("covid_database.db")
cursor = connection.cursor()

# Example query to get confirmed cases, deaths, and recovered cases per country
country = 'USA'
query = f"""
SELECT date, confirmed, deaths, recovered, active
FROM country_wise
WHERE country = '{country}'
"""
cursor.execute(query)
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=['Date', 'Confirmed', 'Deaths', 'Recovered', 'Active'])

# Plot the data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df[['Confirmed', 'Deaths', 'Recovered', 'Active']].plot(figsize=(10, 5))
plt.title(f'COVID-19 Data for {country}')
plt.ylabel('Count')
plt.xlabel('Date')
plt.show()

# Calculate death rate µ(t)
df['DeltaDeaths'] = df['Deaths'].diff().fillna(0)
df['DeathRate'] = df['DeltaDeaths'] / df['Active']

# Assume average recovery time is 4.5 days
recovery_time = 4.5
df['Gamma'] = 1 / recovery_time

# Calculate α(t) and β(t)
df['Alpha'] = (df['Confirmed'].diff().fillna(0) - df['Gamma'] * df['Recovered']) / df['Active']
df['Beta'] = df['Alpha'] / df['Active']

print(df[['DeathRate', 'Gamma', 'Alpha', 'Beta']])

# Trajectory of R0-value
def calculate_r0(df):
    df['R0'] = df['Beta'] / df['Gamma']
    return df['R0']

df['R0'] = calculate_r0(df)
df['R0'].plot(figsize=(10, 5))
plt.title(f'R0 Trajectory for {country}')
plt.ylabel('R0')
plt.xlabel('Date')
plt.show()

# Map of Europe
query = """
SELECT country, active, population
FROM worldometer_data
WHERE continent = 'Europe'
"""
cursor.execute(query)
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=['Country', 'ActiveCases', 'Population'])
df['ActiveCasesPerCapita'] = df['ActiveCases'] / df['Population']

fig = px.choropleth(df, locations='Country', locationmode='country names', color='ActiveCasesPerCapita', scope='europe', title='Active Cases per Capita in Europe')
fig.show()

# Compare Death Rate Over Continents
query = """
SELECT continent, AVG(deaths/active) as avg_death_rate
FROM country_wise
GROUP BY continent
"""
cursor.execute(query)
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=['Continent', 'AvgDeathRate'])

df.plot(kind='bar', x='Continent', y='AvgDeathRate', legend=False)
plt.title('Average Death Rate by Continent')
plt.ylabel('Death Rate')
plt.xlabel('Continent')
plt.show()

# Top 5 US Counties
query = """
SELECT county, SUM(deaths) as total_deaths, SUM(confirmed) as total_cases
FROM usa_county_wise
GROUP BY county
ORDER BY total_deaths DESC
LIMIT 5
"""
cursor.execute(query)
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=['County', 'TotalDeaths', 'TotalCases'])

print(df)

#probeer Odin