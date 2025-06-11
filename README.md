# 🤖 Advanced Automated Trading System

A complete automated trading system in Python integrating stock data retrieval, technical analysis, LSTM prediction, and intelligent trading bots.

## 📊 Real Trading Simulation Results

### 🎯 Conservative vs Aggressive Comparison

We executed real simulations over 30 days for three major stocks:

#### 📈 AAPL (Apple Inc.)
- **Conservative Bot**: 0.00% (0 trades) - No opportunities detected
- **Aggressive Bot**: -1.69% (9 trades) - Success rate: 22.22%
  - Average gain: €46.06
  - Average loss: -€25.24
  - Final capital: €4,915.44

#### 💻 MSFT (Microsoft Corporation)
- **Conservative Bot**: 0.00% (0 trades) - No opportunities detected
- **Aggressive Bot**: +0.02% (1 trade) - Success rate: 100%
  - Average gain: €1.22
  - Average loss: €0.00
  - Final capital: €5,001.22

#### 🚗 TSLA (Tesla Inc.)
- **Conservative Bot**: 0.00% (0 trades) - No opportunities detected
- **Aggressive Bot**: -0.20% (1 trade) - Success rate: 0%
  - Average gain: €0.00
  - Average loss: -€9.87
  - Final capital: €4,990.13

### 📊 Generated Charts

The simulations produced detailed charts for each stock:

- **Price and Trades**: Visualization of entry and exit points
- **Portfolio Evolution**: Capital value tracking
- **Technical Indicators**: RSI, MACD with adaptive thresholds
- **P&L Distribution**: Analysis of gains and losses
- **Performance Summary**: Detailed metrics

### 🔍 Key Observations

1. **Conservative Bot**: Very selective, no trades executed during test period
2. **Aggressive Bot**: More active with permissive parameters
3. **Risk Management**: Automatic stop loss and take profit
4. **Technical Analysis**: Use of RSI, MACD and moving averages

## 🚀 Features

### 📈 Data Collection
- **Yahoo Finance**: Real-time data retrieval
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving averages
- **Historical Data**: Up to 2 years of data

### 🧠 Technical Analysis
- **RSI (Relative Strength Index)**: Overbought/oversold detection
- **MACD**: Moving average convergence/divergence
- **Bollinger Bands**: Volatility and support/resistance levels
- **Moving Averages**: Short and long-term trends

### 🤖 Trading Bots

#### Conservative Bot
- High confidence threshold (0.4)
- Stop loss: 3%
- Take profit: 8%
- Max position: 20% of capital
- Risk per trade: 3%

#### Aggressive Bot
- Low confidence threshold (0.15)
- Stop loss: 1.5%
- Take profit: 4%
- Max position: 30% of capital
- Risk per trade: 5%

### 🧠 LSTM Prediction
- **Ultra-Advanced Model**: Complex LSTM architecture
- **Multiple Features**: Price, volume, technical indicators
- **Multi-Horizon Prediction**: 1, 5, 10 days
- **Backtesting**: Historical data validation

## 📁 Project Structure

```
Share price prediction/
├── src/
│   ├── core/                 # Core modules
│   │   ├── data_collector.py
│   │   ├── technical_analysis.py
│   │   ├── trading_bot_simple.py
│   │   └── lstm_ultra.py
│   ├── demos/               # Demo scripts
│   │   ├── demo_trading_final.py
│   │   ├── demo_trading_aggressive.py
│   │   └── demo_trading_complete.py
│   ├── config/              # Configuration
│   ├── utils/               # Utilities
│   └── main.py              # Main interface
├── images/                  # Generated charts
├── logs/                    # Trading logs
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
pip
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Main Dependencies
- `yfinance`: Stock data
- `pandas`: Data manipulation
- `numpy`: Numerical calculations
- `matplotlib`: Charts
- `scikit-learn`: Machine Learning
- `tensorflow`: LSTM models (optional)

## 🚀 Usage

### Complete Demo
```bash
cd src/demos
python demo_trading_final.py
```

### Conservative vs Aggressive Simulation
```bash
python demo_trading_final.py
```

### Single Stock Simulation
```bash
python demo_trading_final.py single AAPL conservative
python demo_trading_final.py single MSFT aggressive
```

### Main Interface
```bash
cd src
python main.py
```

## 📊 Code Examples

### Technical Analysis
```python
from src.core.data_collector import DataCollector
from src.core.technical_analysis import TechnicalAnalysis

# Get data
collector = DataCollector()
data = collector.get_stock_data('AAPL', period='1y')

# Technical analysis
analyzer = TechnicalAnalysis(data)
analyzer.add_all_indicators()
signals = analyzer.get_signals()
```

### Trading Bot
```python
from src.core.trading_bot_simple import SimpleTradingBot

# Create bot
bot = SimpleTradingBot(
    symbol='AAPL',
    initial_capital=10000,
    risk_per_trade=0.03,
    max_position_size=0.2
)

# Run session
bot.run_trading_session(days=30)
metrics = bot.get_performance_metrics()
```

## 📈 Test Results

### Bot Performance

| Stock | Conservative Bot | Aggressive Bot | Best |
|-------|------------------|----------------|------|
| AAPL  | 0.00% (0 trades) | -1.69% (9 trades) | Conservative |
| MSFT  | 0.00% (0 trades) | +0.02% (1 trade) | Aggressive |
| TSLA  | 0.00% (0 trades) | -0.20% (1 trade) | Conservative |

### Observations
- **Conservative Bot**: Avoids losses but misses opportunities
- **Aggressive Bot**: More activity but risk of losses
- **Risk Management**: Crucial for performance

## 🔧 Configuration

### Conservative Bot Parameters
```python
confidence_threshold = 0.4
stop_loss_pct = 0.03
take_profit_pct = 0.08
max_position_size = 0.2
risk_per_trade = 0.03
```

### Aggressive Bot Parameters
```python
confidence_threshold = 0.15
stop_loss_pct = 0.015
take_profit_pct = 0.04
max_position_size = 0.3
risk_per_trade = 0.05
```

## 📊 Available Charts

### Stock-Specific Charts
- `trading_simulation_AAPL_conservative.png`
- `trading_simulation_AAPL_aggressive.png`
- `trading_simulation_MSFT_conservative.png`
- `trading_simulation_MSFT_aggressive.png`
- `trading_simulation_TSLA_conservative.png`
- `trading_simulation_TSLA_aggressive.png`

### Comparison Charts
- `trading_comparison_final.png`: Complete comparison

## 🧪 Testing

### Unit Tests
```bash
cd tests
python -m pytest
```

### Performance Tests
```bash
python test_trading_bot.py
python test_lstm_predictor.py
```

## 📝 Logs and Monitoring

### Trading Logs
- Files in `logs/`
- Format: `trading_system_YYYYMMDD.log`
- Trade details and performance

### Performance Metrics
- Total return
- Number of trades
- Success rate
- Average gain/loss
- Sharpe ratio

## 🔒 Risk Management

### Automatic Stop Loss
- Protection against significant losses
- Configurable thresholds per bot
- Automatic execution

### Take Profit
- Securing gains
- Adaptive levels
- Return optimization

### Position Sizing
- Exposure limitation
- Risk-based calculation
- Automatic diversification

## 🚀 Future Improvements

### Planned Features
- [ ] Web interface
- [ ] Real-time trading
- [ ] More technical indicators
- [ ] Parameter optimization
- [ ] Advanced backtesting
- [ ] Multi-asset management

### Technical Optimizations
- [ ] Calculation parallelization
- [ ] Data caching
- [ ] Memory optimization
- [ ] REST API

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For any questions or issues:
- Open an issue on GitHub
- Check documentation in `docs/`
- Review logs in `logs/`

---

**⚠️ Disclaimer**: This system is intended for educational and research purposes. Trading involves risk of loss. Use at your own risk. 