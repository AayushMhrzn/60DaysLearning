import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice'],
    'Age': [25, 30, 35, 40, 25, 25],
    'Salary': [50000, 60000, 80000, 90000, 50000, 50000],
    'Department': ['HR', 'IT', 'IT', 'Finance', 'HR', 'HR']
}

df = pd.DataFrame(data)

df['AgePlusTen'] = df['Age'].apply(lambda x: x + 10)
print(df.describe())

df_sorted = df.sort_values(by='Salary', ascending=False)
print(df_sorted)
print(df.corr(numeric_only=True))

print(df['Salary'].idxmin())  # index of minimum salary
print(df['Salary'].idxmax())  # index of maximum salary

print(df['Department'].value_counts())

df['Department'] = df['Department'].replace('HR', 'Human Resources')

print(df['Name'].unique())
print(df['Name'].nunique())

dept_map = {'IT': 'Information Tech', 'Finance': 'Finance Dept', 'Human Resources': 'HR'}
df['DeptFull'] = df['Department'].map(dept_map)

print(df.duplicated())  # Boolean mask
df_no_duplicates = df.drop_duplicates()
print("df with no dups:",df_no_duplicates)