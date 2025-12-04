# Portfolio Analyzer — Monte Carlo Simulation

A financial analysis project for building and optimizing stock portfolios using Monte Carlo simulation.

## 🎯 Project Objectives

* Download historical stock data (S&P 500)
* Compute returns, volatility, and Sharpe Ratio
* Simulate thousands of random portfolios
* Identify the efficient frontier
* Generate visualizations of results

## 📁 Project Structure

```
portfolio_project/
├── data/
│   ├── raw/              # Raw downloaded data
│   ├── cleaned/          # Cleaned data
│   ├── results/          # Simulation results
│   └── plots/            # Generated plots
├── src/
│   ├── data_loader.py    # Data download and cleaning
│   ├── analysis.py       # Financial analysis
│   ├── simulation.py     # Monte Carlo simulation
│   ├── plotting.py       # Visualizations
│   └── utils.py          # Helper functions
├── main.py               # Main script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/enrizani/Portfolio_Analyzer
cd Portfolio_Analyzer
```

### 2. Create a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Full Execution (Download + Analysis + Simulation)

```bash
python3 main.py --full
```

### Use with Existing Data

```bash
python3 main.py
```

### Analysis Only

```bash
python3 main.py --analysis
```

### Simulation Only

```bash
python3 main.py --simulation --n-portfolios 50000
```

### Available Parameters

* `--full`: Complete pipeline including data download
* `--download`: Download new data
* `--clean`: Clean raw data
* `--analysis`: Financial analysis only
* `--simulation`: Monte Carlo simulation only
* `--plots`: Generate plots only
* `--n-portfolios N`: Number of portfolios to simulate (default: 10000)
* `--risk-free R`: Risk-free rate (default: 0.02)
* `--years Y`: Years of historical data (default: 10)
* `--seed S`: Seed for reproducibility (default: 42)
* `--no-show`: Do not display plots

## 📊 Output

### Generated Plots

* `efficient_frontier.png`: Efficient frontier with optimal portfolios
* `optimal_allocations.png`: Optimal portfolio allocations
* `cumulative_returns.png`: Cumulative returns over time
* `correlation_matrix.png`: Correlation matrix
* `returns_distribution.png`: Distribution of returns
* `individual_assets.png`: Performance of individual assets

### Result Files

* `simulation_results.csv`: All simulated portfolios
* `optimal_portfolios.json`: Identified optimal portfolios
* `final_summary.json`: Complete analysis summary

## 🧮 Methodology

### Metrics Calculated

* **Expected Return**: Mean of historical returns (annualized)
* **Volatility**: Standard deviation of returns (annualized)
* **Sharpe Ratio**: (Return − Risk-Free) / Volatility
* **Covariance Matrix**: Relationships between assets

### Monte Carlo Simulation

1. Generate N random portfolios with weights summing to 1
2. Compute expected return and risk for each portfolio
3. Identify optimal portfolios:

   * **Max Sharpe Ratio**
   * **Min Volatility**
   * **Max Return**

### Efficient Frontier

Represents portfolios that provide:

* The highest return for a given level of risk
* The lowest risk for a given level of return

## 📚 Libraries Used

* **pandas** – Data manipulation
* **numpy** – Numerical computations
* **matplotlib** – Plotting
* **seaborn** – Statistical visualization
* **yfinance** – Financial data download

## 🤝 Contributions

Contributions, issues, and feature requests are welcome!

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Your Name**

* GitHub: [@enrizani](https://github.com/enrizani)

## 🙏 Acknowledgements

* Data sourced from Yahoo Finance via *yfinance*
* Inspired by Markowitz’s Modern Portfolio Theory

---

**Note:** This project is for educational purposes only and does not constitute financial advice.
