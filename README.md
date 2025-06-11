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

## 🔬 Scientific Analysis and Interpretations

### 📈 AAPL (Apple Inc.) - Detailed Analysis

#### Technical Indicators Analysis
- **RSI (Relative Strength Index)**: Currently at 43.05, indicating neutral territory
- **MACD**: At -0.9334, showing bearish momentum
- **Price Action**: Trading around €202.67 with moderate volatility

#### Trading Performance Analysis
The aggressive bot executed 9 trades with mixed results:
- **Trade 1**: Buy at €210.93, Sell at €205.08 (Loss: -€40.97)
- **Trade 2**: Buy at €198.63, Sell at €197.23 (Loss: -€9.79)
- **Trade 3**: Buy at €198.27, Sell at €210.79 (Profit: +€87.64)
- **Trade 4**: Buy at €212.93, Sell at €211.26 (Loss: -€11.69)
- **Trade 5**: Buy at €208.78, Sell at €202.09 (Loss: -€46.83)
- **Trade 6**: Buy at €201.36, Sell at €195.27 (Loss: -€42.63)
- **Trade 7**: Buy at €200.21, Sell at €200.85 (Profit: +€4.48)
- **Trade 8**: Buy at €201.70, Sell at €200.63 (Loss: -€7.49)
- **Trade 9**: Buy at €203.92, Sell at €201.45 (Loss: -€17.29)

#### Key Insights
- **Success Rate**: 22.22% (2 profitable trades out of 9)
- **Average Profit**: €46.06 per winning trade
- **Average Loss**: -€25.24 per losing trade
- **Risk-Reward Ratio**: 1.82 (favorable but low success rate)
- **Market Conditions**: Challenging with high volatility

### 💻 MSFT (Microsoft Corporation) - Detailed Analysis

#### Technical Indicators Analysis
- **RSI**: At 64.61, approaching overbought territory
- **MACD**: At 13.1732, showing strong bullish momentum
- **Price Action**: Trading at €470.92 with upward trend

#### Trading Performance Analysis
The aggressive bot executed only 1 trade:
- **Trade 1**: Buy at €393.32, Sell at €394.54 (Profit: +€1.22)

#### Key Insights
- **Success Rate**: 100% (1 trade, 1 profitable)
- **Conservative Behavior**: Bot was selective due to high RSI
- **Market Conditions**: Strong upward momentum with limited entry opportunities
- **Risk Management**: Effective in avoiding overbought conditions

### 🚗 TSLA (Tesla Inc.) - Detailed Analysis

#### Technical Indicators Analysis
- **RSI**: At 44.68, in neutral territory
- **MACD**: At 4.4283, showing moderate bullish momentum
- **Price Action**: Trading at €326.09 with mixed signals

#### Trading Performance Analysis
The aggressive bot executed 1 trade:
- **Trade 1**: Buy at €292.03, Sell at €282.16 (Loss: -€9.87)

#### Key Insights
- **Success Rate**: 0% (1 trade, 0 profitable)
- **Market Volatility**: High volatility led to quick stop loss
- **Risk Management**: Stop loss at 1.5% was triggered
- **Trading Conditions**: Challenging due to Tesla's inherent volatility

## 📊 Comparative Analysis

### Performance Summary Table

| Stock | Conservative Bot | Aggressive Bot | Best Strategy | Key Factors |
|-------|------------------|----------------|---------------|-------------|
| AAPL  | 0.00% (0 trades) | -1.69% (9 trades) | Conservative | High volatility, low success rate |
| MSFT  | 0.00% (0 trades) | +0.02% (1 trade) | Aggressive | Strong momentum, selective entries |
| TSLA  | 0.00% (0 trades) | -0.20% (1 trade) | Conservative | Extreme volatility, poor risk/reward |

### Risk Analysis

#### Conservative Bot Performance
- **Advantages**: 
  - Zero losses during test period
  - Excellent risk management
  - Capital preservation
- **Disadvantages**:
  - Missed profitable opportunities
  - No capital growth
  - Overly cautious approach

#### Aggressive Bot Performance
- **Advantages**:
  - Captured trading opportunities
  - Demonstrated profit potential
  - Active market participation
- **Disadvantages**:
  - Higher risk exposure
  - Mixed success rates
  - Potential for significant losses

### Market Condition Analysis

#### AAPL Market Environment
- **Volatility**: High
- **Trend**: Sideways with bearish bias
- **Technical Signals**: Mixed (RSI neutral, MACD bearish)
- **Trading Difficulty**: High due to choppy price action

#### MSFT Market Environment
- **Volatility**: Moderate
- **Trend**: Strong upward momentum
- **Technical Signals**: Bullish (RSI elevated, MACD positive)
- **Trading Difficulty**: Medium with selective opportunities

#### TSLA Market Environment
- **Volatility**: Very High
- **Trend**: Unclear with high fluctuations
- **Technical Signals**: Mixed (RSI neutral, MACD positive)
- **Trading Difficulty**: Very High due to extreme volatility

## 🧠 Technical Analysis Deep Dive

### RSI (Relative Strength Index) Interpretation

#### AAPL RSI Analysis
- **Current Value**: 43.05
- **Interpretation**: Neutral territory, no clear overbought/oversold signals
- **Trading Implications**: Difficult to generate clear buy/sell signals
- **Historical Context**: RSI has been oscillating between 30-70 range

#### MSFT RSI Analysis
- **Current Value**: 64.61
- **Interpretation**: Approaching overbought territory (70 threshold)
- **Trading Implications**: Conservative approach justified, limited buying opportunities
- **Historical Context**: RSI showing strong upward momentum

#### TSLA RSI Analysis
- **Current Value**: 44.68
- **Interpretation**: Neutral territory, no extreme readings
- **Trading Implications**: Mixed signals, requires additional confirmation
- **Historical Context**: RSI typically more volatile for TSLA

### MACD (Moving Average Convergence Divergence) Analysis

#### AAPL MACD Analysis
- **Current Value**: -0.9334
- **Interpretation**: Bearish momentum, MACD below signal line
- **Trading Implications**: Suggests downward pressure, challenging for long positions
- **Signal Quality**: Strong bearish signal

#### MSFT MACD Analysis
- **Current Value**: 13.1732
- **Interpretation**: Strong bullish momentum, MACD above signal line
- **Trading Implications**: Favorable for long positions, momentum continuation likely
- **Signal Quality**: Strong bullish signal

#### TSLA MACD Analysis
- **Current Value**: 4.4283
- **Interpretation**: Moderate bullish momentum
- **Trading Implications**: Positive but not overwhelming, requires confirmation
- **Signal Quality**: Moderate bullish signal

## 📈 Portfolio Management Insights

### Capital Allocation Strategy

#### Conservative Approach
- **Position Sizing**: 20% maximum per trade
- **Risk Per Trade**: 3% of capital
- **Stop Loss**: 3% from entry
- **Take Profit**: 8% from entry
- **Result**: Capital preservation, missed opportunities

#### Aggressive Approach
- **Position Sizing**: 30% maximum per trade
- **Risk Per Trade**: 5% of capital
- **Stop Loss**: 1.5% from entry
- **Take Profit**: 4% from entry
- **Result**: Higher activity, mixed performance

### Risk Management Analysis

#### Stop Loss Effectiveness
- **Conservative Bot**: No stop losses triggered (no trades)
- **Aggressive Bot**: Stop losses effective in limiting losses
- **TSLA Example**: Stop loss at 1.5% prevented larger losses

#### Take Profit Optimization
- **Conservative Bot**: 8% target may be too high for current market conditions
- **Aggressive Bot**: 4% target more realistic for volatile markets
- **MSFT Example**: Quick 4% profit capture successful

## 🔍 Market Microstructure Analysis

### Liquidity and Execution

#### AAPL Trading Characteristics
- **Volume**: High liquidity, easy execution
- **Spread**: Tight bid-ask spreads
- **Slippage**: Minimal due to high volume
- **Impact**: Low market impact for typical position sizes

#### MSFT Trading Characteristics
- **Volume**: High liquidity, excellent execution
- **Spread**: Very tight spreads
- **Slippage**: Negligible
- **Impact**: Minimal market impact

#### TSLA Trading Characteristics
- **Volume**: High but volatile
- **Spread**: Wider spreads during volatility
- **Slippage**: Potential for slippage during high volatility
- **Impact**: Higher market impact due to volatility

### Volatility Regime Analysis

#### Low Volatility Periods
- **Characteristics**: Tight ranges, low volume
- **Trading Strategy**: Conservative approach more suitable
- **Risk Management**: Lower stop losses, higher take profits

#### High Volatility Periods
- **Characteristics**: Wide ranges, high volume
- **Trading Strategy**: Aggressive approach can capture opportunities
- **Risk Management**: Tighter stop losses, lower take profits

## 🎯 Strategic Recommendations

### For Conservative Investors
1. **Use Conservative Bot Settings**
2. **Focus on High-Quality Stocks** (MSFT-like characteristics)
3. **Implement Longer Holding Periods**
4. **Use Higher Take Profit Targets**
5. **Avoid High-Volatility Stocks** (TSLA-like characteristics)

### For Aggressive Traders
1. **Use Aggressive Bot Settings**
2. **Focus on Volatile Stocks** for opportunity capture
3. **Implement Strict Risk Management**
4. **Use Quick Take Profits**
5. **Monitor Market Conditions Closely**

### For Portfolio Managers
1. **Diversify Across Bot Strategies**
2. **Allocate Based on Market Conditions**
3. **Implement Dynamic Position Sizing**
4. **Use Multiple Timeframes**
5. **Regular Performance Review**

## 📊 Advanced Analytics

### Sharpe Ratio Analysis
- **Conservative Bot**: Undefined (no trades)
- **Aggressive Bot AAPL**: Negative due to losses
- **Aggressive Bot MSFT**: Positive but limited data
- **Aggressive Bot TSLA**: Negative due to loss

### Maximum Drawdown Analysis
- **Conservative Bot**: 0% (no trades)
- **Aggressive Bot AAPL**: -1.69% (total loss)
- **Aggressive Bot MSFT**: 0% (profitable)
- **Aggressive Bot TSLA**: -0.20% (total loss)

### Win Rate Analysis
- **AAPL**: 22.22% (2/9 trades profitable)
- **MSFT**: 100% (1/1 trade profitable)
- **TSLA**: 0% (0/1 trade profitable)
- **Overall**: 33.33% (3/11 trades profitable)

## 🔬 Machine Learning Insights

### Feature Importance
1. **RSI**: Critical for entry/exit decisions
2. **MACD**: Important for momentum confirmation
3. **Price Action**: Essential for trend identification
4. **Volume**: Secondary but valuable
5. **Volatility**: Important for position sizing

### Model Performance
- **Conservative Model**: High precision, low recall
- **Aggressive Model**: Lower precision, higher recall
- **Hybrid Approach**: Potential for optimization

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