import numpy as np
import pandas as pd
import statsmodels.api as sm


#Load the quarterly research dataset.
df = pd.read_csv(
    "data/processed/monetary_credit_risk.csv",
    parse_dates=["observation_date"],
)
df = df.sort_values("observation_date")

#Create the one-quarter lags used by the original macro model.
variables = [
    "delinquency_rate",
    "unemployment_rate",
    "fed_funds_rate",
    "inflation_yoy",
]

for variable in variables:
    df[f"{variable}_lag1"] = df[variable].shift(1)

predictors = [f"{variable}_lag1" for variable in variables]

#Create longer lags for unemployment and the federal funds rate.
lag_columns = []
for variable in ("unemployment_rate", "fed_funds_rate"):
    for lag in (2, 3, 4):
        column = f"{variable}_lag{lag}"
        df[column] = df[variable].shift(lag)
        lag_columns.append(column)

#Use the one-quarter lags to preserve the original split dates.
df = df.dropna(subset=predictors).copy()
split_index = int(len(df) * 0.80)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

validation_index = int(len(train) * 0.80)
development_train = train.iloc[:validation_index]
validation = train.iloc[validation_index:]

print("\n--- Development Data Split ---")
print(f"Training observations: {len(development_train)}")
print(f"Validation observations: {len(validation)}")
print(f"Final test observations: {len(test)}")
print(f"Training ends: {development_train['observation_date'].max().date()}")
print(f"Validation begins: {validation['observation_date'].min().date()}")

#Fit the original one-quarter macro model and evaluate its test forecasts.
X_train = sm.add_constant(train[predictors], has_constant="add")
y_train = train["delinquency_rate"]
model = sm.OLS(y_train, X_train).fit()

X_test = sm.add_constant(test[predictors], has_constant="add")
predictions = model.predict(X_test)
errors = test["delinquency_rate"] - predictions

mae = errors.abs().mean()
rmse = np.sqrt((errors ** 2).mean())

print("\n--- Macroeconomic Regression ---")
print(model.summary())

print("\n--- Out-of-Sample Forecast Evaluation ---")
print(f"Training observations: {len(train)}")
print(f"Testing observations: {len(test)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

#Compare against the test MAE reported by model_baseline.py.
baseline_mae = 0.0381
improvement = (baseline_mae - mae) / baseline_mae * 100
print(f"Improvement over AR(1): {improvement:.2f}%")

#Compare the largest macro forecast errors with naïve persistence.
results = test[
    ["observation_date", "delinquency_rate", "unemployment_rate_lag1"]
].copy()

results["predicted_delinquency"] = predictions
results["absolute_error"] = errors.abs()
results["naive_prediction"] = test["delinquency_rate_lag1"]
results["naive_absolute_error"] = (
    results["delinquency_rate"] - results["naive_prediction"]
).abs()

print("\n--- Eight Largest Macroeconomic Forecast Errors ---")
print(
    results.nlargest(8, "absolute_error").to_string(
        index=False,
        float_format="%.3f",
    )
)

#Measure how much two pandemic quarters contributed to test errors.
pandemic_dates = pd.to_datetime(["2020-07-01", "2020-10-01"])
pandemic_mask = results["observation_date"].isin(pandemic_dates)
remaining_quarters = results.loc[~pandemic_mask]

squared_errors = results["absolute_error"] ** 2
pandemic_share = (
    100 * squared_errors.loc[pandemic_mask].sum() / squared_errors.sum()
)

print("\n--- Pandemic Sensitivity Analysis ---")
print(f"Excluded quarters: {pandemic_mask.sum()}")
print(f"Remaining quarters: {len(remaining_quarters)}")
print(
    f"Macro MAE excluding pandemic quarters: "
    f"{remaining_quarters['absolute_error'].mean():.4f}"
)
print(
    f"Naive MAE excluding pandemic quarters: "
    f"{remaining_quarters['naive_absolute_error'].mean():.4f}"
)
print(f"Pandemic share of total squared error: {pandemic_share:.2f}%")

#Fit all validation models on the same development training rows.
lag_comparison_train = development_train.dropna(subset=lag_columns)


def calculate_validation_mae(model_predictors):
    X_dev = sm.add_constant(
        lag_comparison_train[model_predictors],
        has_constant="add",
    )
    y_dev = lag_comparison_train["delinquency_rate"]
    validation_model = sm.OLS(y_dev, X_dev).fit()

    X_val = sm.add_constant(
        validation[model_predictors],
        has_constant="add",
    )
    predictions_val = validation_model.predict(X_val)

    return (validation["delinquency_rate"] - predictions_val).abs().mean()


naive_validation_mae = (
    validation["delinquency_rate"] - validation["delinquency_rate_lag1"]
).abs().mean()

ar1_validation_mae = calculate_validation_mae(["delinquency_rate_lag1"])

print("\n--- Macro Lag Comparison on Validation Data ---")
print(f"Training observations per fitted model: {len(lag_comparison_train)}")
print(f"Validation observations: {len(validation)}")
print(f"Naive validation MAE: {naive_validation_mae:.4f}")
print(f"AR(1) validation MAE: {ar1_validation_mae:.4f}")

for lag in (1, 2, 3, 4):
    lagged_predictors = [
        "delinquency_rate_lag1",
        f"unemployment_rate_lag{lag}",
        f"fed_funds_rate_lag{lag}",
        "inflation_yoy_lag1",
    ]
    validation_mae = calculate_validation_mae(lagged_predictors)
    print(f"Lag {lag} macro validation MAE: {validation_mae:.4f}")