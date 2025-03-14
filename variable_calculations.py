# Jing
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# Black-Scholes Model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes price for European call or put options.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")


# Implied Volatility Calculation
def implied_volatility(option_price, S, K, T, r, option_type):
    """
    Calculate implied volatility using the Brent root-finding method.
    """

    def objective_function(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - option_price

    try:
        return brentq(objective_function, 1e-6, 10.0, maxiter=1000, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan  # Return NaN if solver fails


# No-Arbitrage Check
def no_arbitrage_check(option_price, S, K, option_type):
    """
    Ensure that the option price follows the no-arbitrage condition.
    """
    if option_type == 'call':
        return option_price >= max(S - K, 0)
    elif option_type == 'put':
        return option_price >= max(K - S, 0)
    return False


# Load and Process Data
file_path = "../i_output/i_complete.csv"
df = pd.read_csv(file_path)

# Ensure required columns exist
required_columns = ['FuturesMidprice', 'OptionsMidprice', 'Strike', 'T', 'type']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Dataset is missing one or more required columns: {required_columns}")

# Filter valid rows based on no-arbitrage condition
df = df[
    df.apply(lambda row: no_arbitrage_check(row['OptionsMidprice'], row['FuturesMidprice'], row['Strike'], row['type']),
             axis=1)]

# Risk-free rate assumption
r = 0.0

# Compute Implied Volatility
df['ImpliedVolatility'] = df.apply(
    lambda row: implied_volatility(
        option_price=row['OptionsMidprice'],
        S=row['FuturesMidprice'],
        K=row['Strike'],
        T=row['T'],
        r=r,
        option_type=row['type']
    ), axis=1
)

# Fill missing volatilities with mean value
mean_volatility = df['ImpliedVolatility'].mean()
df['ImpliedVolatility'].fillna(mean_volatility, inplace=True)

# Save Output
df.to_csv("../i_output/i_implied_volatility2.csv", index=False)
print("Implied volatility calculation completed and saved.")
