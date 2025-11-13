"""
monte_carlo_simulator.py
Monte Carlo Portfolio Simulator (live data by default)
- Fixed tickers: AAPL, MSFT, GOOG, SPY
- LIVE = True -> fetches data from yfinance and saves local CSV snapshot
- LIVE = False -> uses saved CSV "portfolio_data.csv" in repo
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------
# USER SETTINGS (change these)
# -------------------------
LIVE = True               # True => fetch live data; False => use saved CSV
TICKERS = ["AAPL", "MSFT", "GOOG", "SPY"]
START_DATE = "2018-01-01"  # historical window start
END_DATE = None            # None -> uses today's date when LIVE=True
INITIAL_CAPITAL = 10000.0  # used for wealth plot
N_SIMULATIONS = 10000
HORIZON_DAYS = 252         # ~1 trading year
RISK_FREE_RATE = 0.02      # annual risk-free rate used for Sharpe calculation
OUTPUT_PLOTS_DIR = "plots"
DATA_CSV = "portfolio_data.csv"
SEED = 42

np.random.seed(SEED)

# -------------------------
# Helpers: fetch or load data
# -------------------------
def get_price_data(tickers, start=START_DATE, end=END_DATE, live=LIVE):
    """
    Return DataFrame of Adjusted Close prices for tickers.
    If live=True, fetch using yfinance and save CSV snapshot.
    If live=False, read from local CSV file (portfolio_data.csv).
    """
    if live:
        # set end to today string if not provided
        if end is None:
            end = pd.Timestamp.today().strftime("%Y-%m-%d")
        print(f"Downloading data for {tickers} from {start} to {end} ...")
        raw = yf.download(tickers, start=start, end=end, progress=False)
        # yfinance returns columns like ('Adj Close', 'AAPL'); handle both single & multi ticker formats
        if 'Adj Close' in raw.columns:
          prices = raw['Adj Close']
        elif 'Close' in raw.columns:
          prices = raw['Close']
        elif isinstance(raw.columns, pd.MultiIndex) and 'Adj Close' in raw.columns.get_level_values(1):
          prices = raw.xs('Adj Close', axis=1, level=1)
        else:
          raise KeyError("Neither 'Adj Close' nor 'Close' columns found in downloaded data.")

        # Drop columns that are all NaN
        prices = prices.dropna(how="all", axis=1)
        # Save a CSV snapshot for reproducibility (silent fail-safe if permission issue)
        try:
            prices.to_csv(DATA_CSV)
            print(f"Saved price snapshot to {DATA_CSV}")
        except PermissionError:
            print(f"Warning: could not write {DATA_CSV} (PermissionError). Continuing without saving.")
        return prices
    else:
        # Use local saved CSV
        if not os.path.exists(DATA_CSV):
            raise FileNotFoundError(f"{DATA_CSV} not found. Run with LIVE=True at least once to create it.")
        print(f"Loading historical snapshot from {DATA_CSV} ...")
        prices = pd.read_csv(DATA_CSV, index_col=0, parse_dates=True)
        return prices

# -------------------------
# Preprocessing: returns & stats
# -------------------------
def prepare_returns(prices):
    """
    Compute daily simple returns (pct_change) and drop NA rows.
    Returns:
      returns_df (DataFrame): daily returns
      mu (Series): mean daily return per asset
      cov (DataFrame): covariance matrix of daily returns
    """
    # Use simple pct change (you can switch to log returns if you prefer)
    returns = prices.pct_change().dropna(how="all")
    # Align columns to user tickers (safe check)
    returns = returns[TICKERS].dropna(axis=1, how="all")
    mu = returns.mean()         # mean daily return (vector)
    cov = returns.cov()         # daily covariance matrix
    return returns, mu, cov

# -------------------------
# Monte Carlo: random portfolios
# -------------------------
def simulate_random_portfolios(mu, cov, n_portfolios=10000, rf=RISK_FREE_RATE):
    """
    Generate random portfolios (long-only weights that sum to 1).
    Returns a DataFrame with columns: return_annual, vol_annual, sharpe, weights (list)
    """
    n_assets = len(mu)
    results = {
        "return_annual": np.zeros(n_portfolios),
        "vol_annual": np.zeros(n_portfolios),
        "sharpe": np.zeros(n_portfolios),
        "weights": []
    }

    # Annualization factors
    trading_days = 252
    mu_annual = mu * trading_days
    cov_annual = cov * trading_days

    for i in range(n_portfolios):
        # random weights: generate n positive random numbers and normalize to sum to 1
        raw = np.random.random(n_assets)
        weights = raw / np.sum(raw)

        # portfolio expected annual return (dot product)
        port_return = np.dot(weights, mu_annual)

        # portfolio variance and volatility
        port_var = np.dot(weights.T, np.dot(cov_annual.values, weights))
        port_vol = np.sqrt(port_var)

        # Sharpe ratio
        sharpe = (port_return - rf) / port_vol if port_vol != 0 else 0.0

        # Store
        results["return_annual"][i] = port_return
        results["vol_annual"][i] = port_vol
        results["sharpe"][i] = sharpe
        results["weights"].append(weights)

    df = pd.DataFrame({
        "return_annual": results["return_annual"],
        "vol_annual": results["vol_annual"],
        "sharpe": results["sharpe"]
    })
    df["weights"] = results["weights"]
    return df

# -------------------------
# Plotting helpers
# -------------------------
def plot_portfolios(df_portfolios, save_path=None):
    """
    Scatterplot of portfolios: vol (x) vs return (y), colored by sharpe.
    """
    plt.figure(figsize=(10,6))
    sc = plt.scatter(df_portfolios["vol_annual"], df_portfolios["return_annual"],
                     c=df_portfolios["sharpe"], cmap="viridis", s=8, alpha=0.7)
    plt.colorbar(sc, label="Sharpe Ratio")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Random Portfolios: Risk vs Return (color = Sharpe)")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()

def plot_wealth_curve(returns_df, weights, initial_capital=INITIAL_CAPITAL, save_path=None):
    """
    Given historical returns DataFrame and a weight vector, compute the historical wealth curve
    of the portfolio and plot it.
    """
    # align weights with columns order
    weights = np.array(weights)
    # compute portfolio daily returns (value-weighted)
    daily_portfolio_returns = (returns_df * weights).sum(axis=1)
    wealth_series = initial_capital * (1 + daily_portfolio_returns).cumprod()
    plt.figure(figsize=(10,5))
    plt.plot(wealth_series.index, wealth_series.values, label="Portfolio Wealth")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Historical Wealth of Selected Portfolio (backtest)")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()
    return wealth_series

# -------------------------
# Main flow
# -------------------------
def main():
    # 1) Fetch or load price data
    prices = get_price_data(TICKERS, start=START_DATE, end=END_DATE, live=LIVE)
    # Show top rows so you see what data looks like
    print("Price data head:")
    print(prices.head())

    # 2) Preprocess returns & compute mu/cov
    returns_df, mu, cov = prepare_returns(prices)
    print("\nSample mean daily returns (first few):")
    print(mu)
    print("\nSample daily covariance (first few rows):")
    print(cov.iloc[:4,:4])

    # 3) Simulate random portfolios
    print(f"\nSimulating {N_SIMULATIONS} random portfolios ...")
    df_portfolios = simulate_random_portfolios(mu, cov, n_portfolios=N_SIMULATIONS, rf=RISK_FREE_RATE)

    # 4) Find best portfolios
    best_sharpe_idx = df_portfolios["sharpe"].idxmax()
    min_vol_idx = df_portfolios["vol_annual"].idxmin()
    best_sharpe = df_portfolios.loc[best_sharpe_idx]
    min_vol = df_portfolios.loc[min_vol_idx]

    print("\nTop portfolio by Sharpe ratio:")
    print(f"Sharpe = {best_sharpe.sharpe:.3f}, Return = {best_sharpe.return_annual:.3%}, Vol = {best_sharpe.vol_annual:.3%}")
    print("Weights (asset order = {})".format(TICKERS))
    print(np.round(best_sharpe.weights, 4))

    print("\nMinimum volatility portfolio:")
    print(f"Sharpe = {min_vol.sharpe:.3f}, Return = {min_vol.return_annual:.3%}, Vol = {min_vol.vol_annual:.3%}")
    print("Weights (asset order = {})".format(TICKERS))
    print(np.round(min_vol.weights, 4))

    # 5) Plot scatter of portfolios
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    scatter_path = os.path.join(OUTPUT_PLOTS_DIR, "risk_return_scatter.png")
    plot_portfolios(df_portfolios, save_path=scatter_path)
    print(f"Saved portfolio scatter to {scatter_path}")

    # 6) Plot historical wealth curve for the best Sharpe portfolio (backtest)
    wealth_path = os.path.join(OUTPUT_PLOTS_DIR, "wealth_best_sharpe.png")
    wealth_series = plot_wealth_curve(returns_df, best_sharpe.weights, initial_capital=INITIAL_CAPITAL, save_path=wealth_path)
    print(f"Saved wealth plot to {wealth_path}")

    # 7) Basic summary stats for quick README copy/paste
    pct_loss_prob = (wealth_series.iloc[-1] < INITIAL_CAPITAL).mean() if isinstance(wealth_series, pd.Series) else 0.0
    print(f"\nBacktest result on historical window: probability final wealth < initial capital = {pct_loss_prob:.2%}")
    print("Done.")

if __name__ == "__main__":
    main()
