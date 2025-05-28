
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


df = pd.read_csv("/content/Titanic-Dataset.csv")  


print("Basic Information:\n")
print(df.info())
print("\nSummary Statistics:\n")
print(df.describe())
print("\nMissing Values:\n")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns


df[num_cols].hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Features")
plt.show()


plt.figure(figsize=(15, 8))
df[num_cols].boxplot()
plt.title("Boxplots of Numeric Features")
plt.xticks(rotation=90)
plt.show()


sns.pairplot(df[num_cols].sample(200) if len(df) > 200 else df[num_cols])
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

plt.figure(figsize=(12, 8))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

if len(num_cols) >= 2:
    fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title=f'Scatter Plot: {num_cols[0]} vs {num_cols[1]}')
    fig.show()

print("\nFeature-Level Inferences:")

for col in num_cols:
    print(f"Feature: {col}")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Std Dev: {df[col].std():.2f}")
    print("-" * 30)
