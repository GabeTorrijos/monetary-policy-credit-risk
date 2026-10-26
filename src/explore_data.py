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