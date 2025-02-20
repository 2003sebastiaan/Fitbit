import pandas as pd

# Assuming df is your DataFrame containing the daily activity data
df = pd.read_csv('daily_acivity.csv')

# Count the number of occurrences of each Id
id_counts = df['Id'].value_counts()

# Classify each Id based on the number of occurrences
def classify_user(count):
    if count <= 10:
        return 'Light user'
    elif 11 <= count <= 15:
        return 'Moderate user'
    else:
        return 'Heavy user'

# Create a new DataFrame with Id and Class
classification = pd.DataFrame({
    'Id': id_counts.index,
    'Class': id_counts.apply(classify_user)
})

# Reset the index to have a clean DataFrame
classification.reset_index(drop=True, inplace=True)

print(classification)