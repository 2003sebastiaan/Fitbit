import sqlite3
import pandas as pd

connection = sqlite3.connect("fitbit_database.db")
cursor = connection.cursor()

var = 60000
query = "SELECT * FROM employees WHERE salary > ?"
cursor.execute(query, (var,))

rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])


