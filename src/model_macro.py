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
rmse = np.sqrt(np.mean((actual - predictions) **2))

print("\n--- Macroeconomic Regression ---")
print(model.summary())

print("\n--- Out-of-Sample Forecast Evaluation ---")
print(f"Training observations: {len(train)}")
print(f"Testing observations: {len(test)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

#Compare against AR(1) baseline
baseline_mae = 0.0381

improvement = (
    (baseline_mae - mae) / baseline_mae
) * 100

print(f"\nImprovement over AR(1): {improvement:.2f}%")

#Is the macroeconomic model worse across the entire test period or are
#a handful of unusual quarters responsible for most of its errors?

#Examine the largest forecasting errors
results = test[
    [
        "observation_date",
        "delinquency_rate",
        "unemployment_rate_lag1"
    ]
].copy()

results["predicted_delinquency"] = predictions
results["absolute_error"] = (
    results["delinquency_rate"] -
    results["predicted_delinquency"]
).abs()

#Compare errors against naive persistence
results["naive_prediction"] = test["delinquency_rate_lag1"]
results["naive_absolute_error"] = (
    results["delinquency_rate"] -
    results["naive_prediction"]
).abs()

print("\n--- Eight Largest Macroeconomic Forecast Errors ---")

print("Available columns:", results.columns.tolist())

print(
    results.sort_values(
        "absolute_error",
        ascending=False
    ).head(8).round(3).to_string(index=False)
)