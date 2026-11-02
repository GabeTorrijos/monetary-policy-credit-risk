#Get Packages
import pandas as pd
import numpy as np
import statsmodels.api as sm

#Load research dataset 
df = pd.read_csv(
    "data/processed/monetary_credit_risk.csv",
    parse_dates=["observation_date"]
)

#Create one-quarter lagged variables
variables = [
    "delinquency_rate",
    "unemployment_rate",
    "fed_funds_rate",
    "inflation_yoy"
]

for variable in variables:
    df[f"{variable}_lag1"] = df[variable].shift(1)

df = df.dropna().copy()

#Chronological train/test split
split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

#Define predictors
predictors = [
    "delinquency_rate_lag1",
    "unemployment_rate_lag1",
    "fed_funds_rate_lag1",
    "inflation_yoy_lag1"
]

#Train macroeconomic regression
X_train = sm.add_constant(train[predictors])
y_train = train["delinquency_rate"]

model = sm.OLS(y_train, X_train).fit()

#Generate one-quarter-ahead predictions
X_test = sm.add_constant(
    test[predictors],
    has_constant="add"
)

predictions = model.predict(X_test)
actual = test["delinquency_rate"]

#Calculate forecast errors
mae = np.mean(np.abs(actual - predictions))
rsme = np.sqrt(np.mean((actual - predictions) **2))
