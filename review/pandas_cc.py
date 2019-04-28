import pandas as pd

df = pd.read_csv('../__tensorflow__/00-Crash-Course-Basics/salaries.csv')
print(df)
print(df['Salary'])
print(df[['Salary', 'Name']])
print(df['Salary'].max())
print(df.describe())
print(df[df['Salary'] > 60000])
