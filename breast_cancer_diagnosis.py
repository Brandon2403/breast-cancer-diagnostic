import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Your Dataset
df = pd.read_csv('data.csv')

# Convert 'diagnosis' column to numeric values (0 for 'B' and 1 for 'M')
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['float64']).columns

# Compute correlation matrix for numeric columns only
correlation_matrix = df[numeric_columns].corr()

# Print the correlation matrix
print(correlation_matrix)

# Explore the First Few Rows
print(df.head())

# Check Basic Information
print(df.info())

# Descriptive Statistics
print(df.describe())

# Correlation Analysis
correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

