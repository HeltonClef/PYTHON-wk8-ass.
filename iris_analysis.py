
# Iris Dataset Analysis Script
# Task 1: Load and Explore the Dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
try:
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check structure: data types and missing values
print("\nDataset info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# Clean dataset (not needed here since Iris has no missing values)
df.dropna(inplace=True)

# Task 2: Basic Data Analysis

# Basic statistics
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Grouping by species and getting mean of numerical columns
grouped = df.groupby("target").mean()
print("\nMean values grouped by species (target):")
print(grouped)

# Mapping target numbers to species names for clarity
df['species'] = df['target'].map(dict(enumerate(iris_data.target_names)))
print("\nFirst 5 rows with species name:")
print(df[['target', 'species']].head())

# Insight: Print which species has the highest average petal length
max_petal_length_species = df.groupby("species")["petal length (cm)"].mean().idxmax()
print(f"\nSpecies with highest average petal length: {max_petal_length_species}")

# Task 3: Data Visualization

# Set Seaborn theme
sns.set(style="whitegrid")

# 1. Line chart (artificial time-series by row index)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["sepal length (cm)"], label='Sepal Length')
plt.plot(df.index, df["sepal width (cm)"], label='Sepal Width')
plt.title("Line Chart of Sepal Length & Width Over Index")
plt.xlabel("Index")
plt.ylabel("cm")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart: average petal length per species
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="species", y="petal length (cm)", palette="pastel")
plt.title("Average Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.tight_layout()
plt.show()

# 3. Histogram of sepal length
plt.figure(figsize=(7, 5))
plt.hist(df["sepal length (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter plot: sepal vs petal length
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="deep")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()
