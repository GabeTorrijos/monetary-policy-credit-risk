#Load Packages
import pandas as pd

#Load cleaned research dataset
df = pd.read_csv(
    "data/processed/monetary_credit_risk.csv",
    parse_dates=["observation_date"]
)

#Display basic information
print("\n--- Research Dataset ---")
print("Observations:", len(df))
print("First Quarter:", df["observation_date"].min())
print("Last Quarter:", df["observation_date"].max())

#Calculate descriptive statistics
print("\n--- Descriptive Statistics ---")

variables = [
    "delinquency_rate",
    "fed_funds_rate",
    "unemployment_rate",
    "inflation_yoy"
]

print(df[variables].describe().round(2).to_string())