#Get packages

import pandas as pd

#Load the four raw FRED datasets
delinquency = pd.read_csv("data/raw/commercial_bank_delinquency.csv")
fed_funds = pd.read_csv("data/raw/federal_funds_rate.csv")
unemployment = pd.read_csv("data/raw/unemployment_rate.csv")
cpi = pd.read_csv("data/raw/consumer_price_index.csv")

#Display the dimensions of each dataset
print("Commercial Bank Delinquency:", delinquency.shape)
print("Federal Funds Rate:", fed_funds.shape)
print("Unemployment Rate:", unemployment.shape)
print("Consumer Price Index", cpi.shape)

#Inspect the structure of each dataset
datasets = { 
    "Commercial Bank Delinquency": delinquency,
    "Federal Funds Rate": fed_funds,
    "Unemployment Rate": unemployment,
    "Consumer Price Index": cpi
}

for name, df in datasets.items():
    print(f"\n--- {name} ---")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

 
 # Convert observation dates to datetime objects
for name, df in datasets.items():
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    print(f"\n{name}")
    print(df.dtypes)

#Merge datasets using their observation dates
merged = delinquency.merge(
    fed_funds,
    on="observation_date",
    how="inner"
)

merged = merged.merge(
    unemployment,
    on="observation_date",
    how="inner"
)

merged = merged.merge(
    cpi,
    on="observation_date",
    how="inner"
)

#Inspect the combined dataset
print("\n--- Combined Economic Dataset ---")
print("Dimensions:", merged.shape)
print("\nFirst Five Observations:")
print(merged.head())

print("\nMissing Values:")
print(merged.isnull().sum())

#Calculate year-over-year inflation from quarterly CPI
merged = merged.sort_values("observation_date").reset_index(drop=True)

merged["inflation_yoy"] = (
    merged["CPIAUCSL"].pct_change(periods=4) * 100
)

#Inspect the inflation calculation
print("\n--- Inflation Calculation ---")
print(
    merged[
        ["observation_date", "CPIAUCSL", "inflation_yoy"]
    ].head(8)
)

print("\nMissing Inflation Values:")
print(merged["inflation_yoy"].isnull().sum())

#Remove observations with missing inflation data
cleaned = merged.dropna(subset=["inflation_yoy"]).copy()

#Select and rename variables for research dataset
cleaned = cleaned[
    [
        "observation_date",
        "DRALACBS",
        "FEDFUNDS",
        "UNRATE",
        "inflation_yoy"
    ]
].rename(
    columns={
        "DRALACBS": "delinquency_rate",
        "FEDFUNDS": "fed_funds_rate",
        "UNRATE": "unemployment_rate"
    }
)

#Save the cleaned dataset
cleaned.to_csv("data/processed/monetary_credit_risk.csv", index=False)

#Verify the final dataset
print("\n--- Clean Research Dataset ---")
print("Dimensions:", cleaned.shape)
print("\nFirst Five Observations:")
print(cleaned.head())

print("\nMissing Values:")
print(cleaned.isnull().sum())