import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [24, 30, 22, 25, 29],
    'City': ['Kathmandu', 'Lalitpur', 'Bhaktapur', 'Kathmandu', 'Pokhara']
}

df = pd.DataFrame(data)

# Filter rows where Age > 25
print(df[df['Age'] > 25])

# Filter rows where City is either Kathmandu or Pokhara
print(df[df['City'].isin(['Kathmandu', 'Pokhara'])])

# Filter rows where Age is between 23 and 29
print(df[df['Age'].between(23, 29)])

# Filter rows where City contains 'pur'
print(df[df['City'].str.contains('pur')]) #.str.contains('pur', case=False) for case insensitive

# Same as Age > 25 and City == 'Kathmandu'
print(df.query("Age < 25 and City == 'Kathmandu'"))

# Select rows where Age > 25 and only Name & City columns
print(df.loc[df['Age'] > 25, ['Name', 'City']])

# Select columns that start with 'C'
print(df.filter(regex='^C'))  # Matches 'City'
