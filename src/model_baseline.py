#Get packages
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

#Load dataset
df = pd.read_csv(
    "data/processed/monetary_credit_risk.csv",
    parse_dates=["observation_date"]
)

#Create lagged delinquency
df["delinquency_lag1"] = df["delinquency_rate"].shift(1)
df = df.dropna().copy()

#Chronological train/test split
split_index = int(len(df) * 0.80)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

#Define training variables
X_train = sm.add_constant(train[["delinquency_lag1"]])
y_train = train["delinquency_rate"]

#Fit regression using training data only
model = sm.OLS(y_train, X_train).fit()

#Generate test predictions
X_test = sm.add_constant(
    test[["delinquency_lag1"]],
    has_constant="add"
)

predictions = model.predict(X_test)

#Calculate forcasting errors
mae = mean_absolute_error(test["delinquency_rate"], predictions)
rmse = np.sqrt(
    mean_squared_error(test["delinquency_rate"], predictions)
)

print("\n--- AR(1) Out-of-Sample Forecast Evaluation ---")
print(f"Training observations: {len(train)}")
print(f"Testing observation: {len(test)}")
print(f"Training period: {train['observation_date'].min().date()} to {train['observation_date'].max().date()}")
print(f"Testing period: {test['observation_date'].min().date()} to {test['observation_date'].max().date()}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")