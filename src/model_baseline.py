#Get packages
import pandas as pd
import statsmodels.api as sm

#Load research dataset
df = pd.read_csv(
    "data/processed/monetary_credit_risk.csv",
    parse_dates=["observation_date"]
)

#Create previous-quarter delinquency variable
df["delinquency_lag1"] = df["delinquency_rate"].shift(1)

#Remove first observation (no previous-quarter data)
df = df.dropna().copy()

#Define dependent and independent variables
y = df["delinquency_rate"]
X = df[["delinquency_lag1"]]

#Add regression intercept
X = sm.add_constant(X)

#Estimate AR(1) regression
model = sm.OLS(y, X).fit()

#Display results
print("\n--- AR(1) Baseline Regression ---")
print(model.summary())