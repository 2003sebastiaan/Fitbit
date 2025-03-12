import sqlite3
import pandas as pd

# Connect to the fitbit database
connection = sqlite3.connect("fitbit_database.db")
cursor = connection.cursor()

# List all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)

# Check the structure of each table to find relevant data
for table in tables:
    table_name = table[0]
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print(f"\nColumns in {table_name} table:")
    for column in columns:
        print(column)

# Query to get the weight log data
query = """
SELECT *
FROM weight_log
"""
cursor.execute(query)

rows = cursor.fetchall()
weight_log_df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])

# Print the entire weight_log table
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
print("\nWeight log table:")
print(weight_log_df)

# Fill missing values in WeightKg using WeightPounds
weight_log_df['WeightKg'] = weight_log_df.apply(
    lambda row: row['WeightKg'] if pd.notnull(row['WeightKg']) else row['WeightPounds'] * 0.453592,
    axis=1
)

# Calculate missing Fat values using WeightKg and BMI
def calculate_fat(row):
    if pd.notnull(row['Fat']):
        return row['Fat']
    elif pd.notnull(row['WeightKg']) and pd.notnull(row['BMI']):
        # Example formula to estimate body fat percentage
        # This is a simplified formula and may not be accurate for all individuals
        return (1.20 * row['BMI']) + (0.23 * 30) - 5.4  # Assuming age is 30 for this example
    else:
        return None

weight_log_df['NewFat'] = weight_log_df.apply(calculate_fat, axis=1)

# Print the updated weight_log table
print("\nUpdated weight_log table with missing WeightKg and NewFat values filled:")
print(weight_log_df)

# Add new columns to the database
cursor.execute("ALTER TABLE weight_log ADD COLUMN NewWeightKg REAL")
cursor.execute("ALTER TABLE weight_log ADD COLUMN NewFat REAL")

# Update the new columns with the calculated values
for index, row in weight_log_df.iterrows():
    cursor.execute("""
    UPDATE weight_log
    SET NewWeightKg = ?, NewFat = ?
    WHERE Id = ? AND Date = ?
    """, (row['WeightKg'], row['NewFat'], row['Id'], row['Date']))

# Commit the changes
connection.commit()

# Save the cleaned data to a new CSV file
weight_log_df.to_csv('cleaned_weight_log.csv', index=False)

print("Data wrangling complete. Cleaned data saved to 'cleaned_weight_log.csv'.")

