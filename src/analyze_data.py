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

#Load packages
import matplotlib.pyplot as plt

#Plot historical commercial bank delinquency rates
plt.figure(figsize=(12, 6))

plt.plot(
    df["observation_date"],
    df["delinquency_rate"],
    color="navy",
    linewidth=2
)

plt.title(
    "U.S. Commercial Bank Loan Delinquency Rate(1986-2026)",
    fontsize=14
)

plt.xlabel("Year")
plt.ylabel("Delinquency Rate (%)")

plt.grid(True, alpha=0.3)
plt.tight_layout()

#Save visualization
plt.savefig(
    "outputs/figures/delinquency_history.png",
    dpi=300
)

plt.show()

#Compare delinquency rates and unemployment over time
plt.figure(figsize=(12, 6))

plt.plot(
    df["observation_date"],
    df["delinquency_rate"],
    label="Loan Delinquency Rate",
    color="navy",
    linewidth=2
)

plt.plot(
    df["observation_date"],
    df["unemployment_rate"],
    label="Unemployment Rate",
    color="darkred",
    linewidth=2,
    alpha=0.8
)

plt.title(
    "Commercial Bank Delinquency vs. Unemployment (1986-2026)",
    fontsize=14
)

plt.xlabel("Year")
plt.ylabel("Rate (%)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/figures/delinquency_vs_unemployment.png",
    dpi=300
)

plt.show()

#Calculate correlation between economic indicators
correlation_matrix = df[variables].corr()

print("\n--- Correlation Matrix ---")
print(correlation_matrix.round(3).to_string())