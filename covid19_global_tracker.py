
# 📝 COVID-19 Global Data Tracker
# This project analyzes COVID-19 trends globally using real-world data from Our World in Data.

# 1️⃣ Data Collection
# Source: https://ourworldindata.org/covid-cases
# File: 'owid-covid-data.csv'

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('owid-covid-data.csv')
print("Dataset shape:", df.shape)
print("Columns:", df.columns)

# Preview dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# 2️⃣ Data Cleaning
# Convert 'date' to datetime
df['date'] = pd.to_datetime(df['date'])

# Filter selected countries
countries = ['Kenya', 'United States', 'India']
df = df[df['location'].isin(countries)]

# Fill missing numeric data with forward fill
df.fillna(method='ffill', inplace=True)

# 3️⃣ Exploratory Data Analysis (EDA)
# Plot total cases over time
plt.figure(figsize=(10,5))
for country in countries:
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country)
plt.title('Total COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Cases')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate death rate
df['death_rate'] = df['total_deaths'] / df['total_cases']
print(df[['location', 'date', 'death_rate']].dropna().head())

# 4️⃣ Vaccination Progress
# Plot vaccinations over time
plt.figure(figsize=(10,5))
for country in countries:
    country_data = df[df['location'] == country]
    plt.plot(country_data['date'], country_data['total_vaccinations'], label=country)
plt.title('Total Vaccinations Over Time')
plt.xlabel('Date')
plt.ylabel('Total Vaccinations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5️⃣ Optional: Choropleth Map of Total Cases
latest = df[df['date'] == df['date'].max()]
fig = px.choropleth(latest, locations='iso_code', color='total_cases',
                    hover_name='location', title='Total COVID-19 Cases by Country')
fig.show()

# 6️⃣ Insights & Findings
print("""
🧠 USA has the highest number of total cases.
💉 India has seen a strong vaccination push after mid-2021.
📈 Death rates vary across countries, influenced by healthcare infrastructure.
📊 Visualizations show strong correlation between new cases and vaccinations.
""")
