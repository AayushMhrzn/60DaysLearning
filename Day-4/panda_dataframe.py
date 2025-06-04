import pandas as pd

# Creating a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [24, 30, 22],
    'City': ['Kathmandu', 'Lalitpur', 'Bhaktapur']
}

df = pd.DataFrame(data)
print(df)
print(df.shape)          # Rows and columns
print(df.columns)        # Column names
print(df.info())         # Column data types and non-null info
print(df.describe())     # Stats for numeric columns

# Select a column
print(df['Name'])  

# Select multiple columns
print(df[['Name', 'City']])

# Select row by index
print(df.loc[1])  # Label-based
print(df.iloc[1]) # Position-based

# Select value at row 0, column 'City'
print(df.at[0, 'City'])

# View index
print(df.index)

# Set a new index
df2 = df.set_index('Name')
print(df2)

# Reset index
df2 = df2.reset_index()
print(df2)

