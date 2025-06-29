# 📈 BOST - Backtesting and Optimization System for Trading

<div align="center">

![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**A powerful, user-friendly platform for backtesting and optimizing trading strategies**

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [Strategies](#-strategies)

</div>

---

## 🎯 Overview

BOST is a comprehensive backtesting and optimization platform designed for traders, quantitative analysts, and financial researchers. Built with Python and powered by Streamlit, it provides an intuitive interface for testing trading strategies against historical data, optimizing parameters, and analyzing performance with professional-grade metrics.

Whether you're a beginner exploring algorithmic trading or an experienced quant developing sophisticated strategies, BOST offers the tools you need to validate your ideas and improve your trading performance.

## 🚀 Features

### 📊 **Comprehensive Backtesting**
- Simulate strategies against historical market data with precision
- Support for any asset class (stocks, forex, crypto, commodities)
- Multiple timeframe analysis capabilities
- Real-time performance tracking

### ⚡ **Robust Optimization**
- Advanced parameter optimization algorithms
- Grid search and random search methods
- Multi-objective optimization support
- Parallel processing for faster results

### 📈 **Advanced Analytics**
- **Performance Metrics**: Total Return, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, SQN
- **Risk Analysis**: Value at Risk (VaR), volatility measures, correlation analysis
- **Statistical Tests**: Monte Carlo simulations for strategy robustness
- **Benchmark Comparison**: Compare against market indices and other strategies

### 🎨 **Interactive Visualization**
- Dynamic charts and performance graphs
- Equity curve analysis
- Drawdown visualization
- Trade distribution analysis
- Risk-return scatter plots

### 📋 **Professional Reporting**
- Detailed Excel export functionality
- Comprehensive performance reports
- Strategy comparison matrices
- Risk assessment summaries

### 🏗️ **Modular Architecture**
- Easy strategy implementation
- Flexible data handling
- Extensible framework
- Clean, organized codebase

### 💻 **User-Friendly Interface**
- Streamlit-powered GUI
- No coding required for basic usage
- Integrated data sources
- Real-time parameter adjustment

## 🛠️ Installation

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BOST-Backtesting-and-Optimization.git

# Navigate to the project directory
cd BOST-Backtesting-and-Optimization

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Launch the Application
```bash
streamlit run main.py
```

### 2. Basic Workflow
1. **Load Data**: Use integrated data functions (no external files needed)
2. **Select Strategy**: Choose from pre-built strategies or create your own
3. **Configure Parameters**: Set your strategy parameters and backtest settings
4. **Run Backtest**: Execute the backtest and view real-time results
5. **Optimize**: Fine-tune parameters for optimal performance
6. **Analyze Results**: Explore interactive charts and detailed metrics
7. **Export Reports**: Generate comprehensive Excel reports

## 📖 Usage

### Data Integration
BOST features fully integrated data handling:
- Built-in data management system
- No external data files required
- Multiple timeframes
- Support for any financial instrument

<!-- ## 📊 Screenshots

*This section showcases the BOST interface and key features*

<!-- Screenshots will be added here -->
<!-- Dashboard Interface -->
<!-- Strategy Results Visualization -->
<!-- Optimization Interface -->
<!-- Performance Charts and Analytics -->

## 🎯 Strategies

BOST comes with a comprehensive collection of pre-built strategies:

### Technical Indicators
- **Moving Average Crossover** (`strategy_ma_crossover.py`) - Classic dual MA crossover system
- **RSI Strategy** (`strategy_rsi.py`) - Relative Strength Index based trading
- **RSI Multi-Timeframe** (`strategy_rsi_multi_timeframe.py`) - Multi-timeframe RSI analysis
- **RSI Price Divergence** (`strategy_rsi_price_divergence.py`) - Divergence detection system
- **MACD Strategy** (`strategy_macd.py`) - Moving Average Convergence Divergence
- **Bollinger Bands** (`strategy_bollinger_bands.py`) - Volatility-based trading

### Price Action
- **Breakout Strategy** (`strategy_breakout.py`) - Price breakout detection

### Benchmark
- **Buy and Hold** (`buy_and_hold_strategy.py`) - Passive investment benchmark

### Custom Strategies
Easily create your own strategies using the `common_strategy.py` base class framework.

## 📁 Project Structure

```
BOST/
├── .streamlit/                      # Streamlit configuration
│   └── config.toml
├── .vscode/                         # VS Code settings
│   ├── settings.json
│   └── launch.json
├── strategies/                      # Trading strategies directory
│   ├── __init__.py
│   ├── common_strategy.py          # Base strategy class
│   ├── strategy_ma_crossover.py    # Moving Average strategy
│   ├── strategy_rsi.py             # RSI strategy
│   ├── strategy_rsi_multi_timeframe.py
│   ├── strategy_rsi_price_divergence.py
│   ├── strategy_macd.py            # MACD strategy
│   ├── strategy_bollinger_bands.py # Bollinger Bands strategy
│   ├── strategy_breakout.py        # Breakout strategy
│   └── buy_and_hold_strategy.py    # Buy and hold benchmark
├── main.py                         # Main application entry point
├── ui.py                          # Streamlit user interface
├── backtest_runner.py             # Backtesting engine
├── optimizer_runner.py            # Optimization engine
├── data_handler.py                # Data management system
├── display_results.py             # Results visualization
├── excel_exporter.py              # Excel export functionality
├── monte_carlo.py                 # Monte Carlo simulations
├── utils.py                       # Utility functions
├── config.py                      # Configuration settings
├── messages.yaml                  # UI messages and labels
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── .gitignore                     # Git ignore rules
```

## 📊 Performance Metrics

BOST calculates comprehensive performance metrics including:

### Return Metrics
- **Total Return** - Overall strategy performance
- **Annualized Return** - Yearly performance average
- **Excess Return** - Performance vs benchmark

### Risk Metrics
- **Sharpe Ratio** - Risk-adjusted return measure
- **Sortino Ratio** - Downside risk-adjusted return
- **Maximum Drawdown** - Largest peak-to-trough decline
- **System Quality Number (SQN)** - Trading system quality assessment

### Trading Metrics
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Ratio of gross profit to gross loss
- **Average Trade** - Mean trade performance
- **Trade Distribution** - Statistical analysis of trade outcomes

### Key Dependencies
- **pandas** - Data manipulation and analysis
- **numpy** - Scientific computing and numerical operations
- **streamlit** - Web application framework
- **backtesting.py** - Core backtesting engine
- **matplotlib** - Plotting and visualization
<!-- ## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 BOST Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
-->
## 🙏 Acknowledgments

- **[Backtesting.py](https://kernc.github.io/backtesting.py/)** - Core backtesting engine
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** - Scientific computing
- **[Matplotlib](https://matplotlib.org/)** - Plotting and visualization

---

<div align="center">

**Made with ❤️ for the trading community by someone who is not a programmer**

⭐ **Star this repository if you find it useful!** ⭐

</div>