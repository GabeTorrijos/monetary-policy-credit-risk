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

#Create training and validation sets
validation_index = int(len(train) * 0.80)

development_train = train.iloc[:validation_index].copy()
validation = train.iloc[validation_index:].copy()

print("\n--- Development Data Split ----")
print(f"Training observations: {len(development_train)}")
print(f"Validation observations: {len(validation)}")
print(f"Final test observations: {len(test)}")

print(
    f"Training ends: "
    f"{development_train['observation_date'].max().date()}"
)

print(
    f"Validation begins: "
    f"{validation['observation_date'].min().date()}"
)

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
    ).head(8).to_string(index=False, float_format="%.3f")
)

#Sensitivity analysis: pandemic forecast errors
pandemic_dates = pd.to_datetime([
    "2020-07-01",
    "2020-10-01"
])

pandemic_mask = results["observation_date"].isin(pandemic_dates)

normal_period = results.loc[~pandemic_mask]

#Calculate contribution to total squared error
squared_errors = results["absolute_error"] ** 2

pandemic_share = (
    squared_errors.loc[pandemic_mask].sum()
    / squared_errors.sum()
) * 100

print("\n--- Pandemic Sensitivity Analysis ---")
print(f"Excluded quarters: {pandemic_mask.sum()}")
print(f"Remaining quarters: {len(normal_period)}")

print(
    f"Macro MAE excluding pandemic quarters: "
    f"{normal_period['absolute_error'].mean():.4f}"
)

print(
    f"Naive MAE excluding pandemic quarters: "
    f"{normal_period['naive_absolute_error'].mean():.4f}"
)

print(
    f"Pandemic share of total squared error: "
    f"{pandemic_share:.2f}%"
)

#Compare baseline and macro models on validation data
print("\n--- Validation Model Comparison ---")

def calculate_validation_mae(predictors):
    X_dev = sm.add_constant(
        development_train[predictors],
        has_constant="add"
    )

    y_dev = development_train["delinquency_rate"]

    validation_model = sm.OLS(y_dev, X_dev).fit()

    X_val = sm.add_constant(
        validation[predictors],
        has_constant="add"
    )

    predictions_val = validation_model.predict(X_val)

    return np.mean(
        np.abs(validation["delinquency_rate"] - predictions_val)
    )


baseline_validation_mae = calculate_validation_mae(
    ["delinquency_rate_lag1"]
)

macro_validation_mae = calculate_validation_mae(
    predictors
)

print(f"AR(1) Validation MAE: {baseline_validation_mae:.4f}")
print(f"Macro Validation MAE: {macro_validation_mae:.4f}")

#Naive forecast on validation data 
naive_validation_mae = np.mean(
    np.abs(
        validation["delinquency_rate"] -
        validation["delinquency_rate_lag1"]
    )
)

print("\n--- Naive Validation Benchmark ---")
print(f"Naive Validation MAE: {naive_validation_mae:.4f}")
print(f"AR(1) Validation MAE: {baseline_validation_mae:.4f}")
print(f"Macro Validation MAE: {macro_validation_mae:.4f}")