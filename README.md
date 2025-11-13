# 🧮 Portfolio Optimization using Modern Portfolio Theory

Simulate and optimize portfolio allocations using **random portfolio generation**, **Sharpe ratio maximization**, and **minimum volatility selection**. Includes visualization of efficient frontiers and wealth growth.


## 💡 Why this project

- Implements a **core quant finance concept**: risk–return optimization under Modern Portfolio Theory (MPT).  
- Demonstrates **practical skills** in Python, NumPy, Pandas, and Matplotlib for financial analysis.  
- Features a **toggle for live vs. static data**, showcasing reproducible research design.  


## ⚙️ Features

- Pulls historical prices (AAPL, MSFT, GOOG, SPY) using `yfinance`.  
- Simulates thousands of random portfolios and computes:
  - Portfolio **expected return**
  - Portfolio **volatility**
  - Portfolio **Sharpe ratio**  
- Identifies:
  - ✅ **Best Sharpe ratio portfolio**
  - ✅ **Minimum volatility portfolio**
- Visualizes:
  - Efficient frontier (risk–return scatter)
  - Wealth growth of top portfolio
- Supports **live data (current market)** or **snapshot (saved CSV)** toggle.


## 📂 Project Structure

portfolio_optimization/
│
├── portfolio_optimizer.py # main script
├── requirements.txt # dependencies
├── README.md # project documentation
├── plots/ # auto-generated charts
└── .gitignore # ignore cache/data files


## 🧭 Installation

```bash
# Clone the repo
git clone https://github.com/krishna-quant/portfolio-optimization.git
cd portfolio-optimization

# Install dependencies
pip install -r requirements.txt


## 🚀 Usage

1️⃣ Run with snapshot data
Live = False
python portfolio_optimizer.py

2️⃣ Run with live data
Live = True
python portfolio_optimizer.py

The script will:

- Fetch or read data

- Simulate 10,000 random portfolios

= Display best Sharpe and min-vol portfolios

- Save risk–return and wealth plots under /plots


## 📊 Example Output

Top portfolio by Sharpe ratio:
Sharpe = 0.964, Return = 27.839%, Vol = 26.809%
Weights = [AAPL=39%, MSFT=44%, GOOG=17%, SPY=0.04%]

## 🧠 Skills Demonstrated

- Financial data analysis

- Portfolio simulation & optimization

- Python for quantitative finance

- Data visualization & reproducibility

## 🪪 License

MIT



