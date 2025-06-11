"""
Simplified analysis script for generating charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.trading_bot_simple import SimpleTradingBot
from core.data_collector import DataCollector
from core.technical_analysis import TechnicalAnalysis

# Set style
plt.style.use('default')
sns.set_palette("husl")

def generate_stock_analysis(symbol: str):
    """
    Generate comprehensive stock analysis with visualizations
    """
    print(f"📊 Generating analysis for {symbol}...")
    
    # Fetch data
    collector = DataCollector()
    data = collector.get_stock_data(symbol, period="1y")
    if data is None:
        return {}
    
    # Technical analysis
    analyzer = TechnicalAnalysis(data)
    analyzer.add_all_indicators()
    analyzed_data = analyzer.data
    
    # Create visualizations
    create_price_chart(symbol, analyzed_data)
    create_technical_indicators(symbol, analyzed_data)
    create_performance_analysis(symbol, analyzed_data)
    
    return {
        'symbol': symbol,
        'current_price': analyzed_data['Close'].iloc[-1],
        'total_return': (analyzed_data['Close'].iloc[-1] / analyzed_data['Close'].iloc[0] - 1) * 100,
        'volatility': analyzed_data['Close'].pct_change().std() * np.sqrt(252) * 100
    }

def create_price_chart(symbol: str, data: pd.DataFrame):
    """
    Create price chart with technical indicators
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Main price chart
    ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='#2E86AB')
    
    # Moving averages
    if 'SMA_20' in data.columns:
        ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7, color='#A23B72')
    if 'SMA_50' in data.columns:
        ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7, color='#F18F01')
    if 'SMA_200' in data.columns:
        ax1.plot(data.index, data['SMA_200'], label='SMA 200', alpha=0.7, color='#C73E1D')
    
    # Bollinger Bands
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], 
                        alpha=0.2, color='gray', label='Bollinger Bands')
    
    ax1.set_title(f'{symbol} Stock Price Analysis with Technical Indicators', 
                 fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Volume chart
    ax2.bar(data.index, data['Volume'], alpha=0.6, color='#4ECDC4', label='Volume')
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/analysis_{symbol}_price_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Price chart saved: ../images/analysis_{symbol}_price_chart.png")

def create_technical_indicators(symbol: str, data: pd.DataFrame):
    """
    Create technical indicators visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # RSI
    if 'RSI' in data.columns:
        ax1.plot(data.index, data['RSI'], color='#FF6B6B', linewidth=2)
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax1.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax1.set_title('Relative Strength Index (RSI)', fontweight='bold')
        ax1.set_ylabel('RSI')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # MACD
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        ax2.plot(data.index, data['MACD'], label='MACD', color='#4ECDC4', linewidth=2)
        ax2.plot(data.index, data['MACD_Signal'], label='Signal', color='#FFE66D', linewidth=2)
        if 'MACD_Histogram' in data.columns:
            ax2.bar(data.index, data['MACD_Histogram'], alpha=0.5, color='#95A5A6', label='Histogram')
        ax2.set_title('MACD (Moving Average Convergence Divergence)', fontweight='bold')
        ax2.set_ylabel('MACD')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Stochastic Oscillator
    if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
        ax3.plot(data.index, data['Stoch_K'], label='%K', color='#9B59B6', linewidth=2)
        ax3.plot(data.index, data['Stoch_D'], label='%D', color='#3498DB', linewidth=2)
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax3.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax3.set_title('Stochastic Oscillator', fontweight='bold')
        ax3.set_ylabel('Stochastic')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Williams %R
    if 'Williams_R' in data.columns:
        ax4.plot(data.index, data['Williams_R'], color='#E74C3C', linewidth=2)
        ax4.axhline(y=-20, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax4.axhline(y=-80, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax4.set_title('Williams %R', fontweight='bold')
        ax4.set_ylabel('Williams %R')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/analysis_{symbol}_technical_indicators.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Technical indicators saved: ../images/analysis_{symbol}_technical_indicators.png")

def create_performance_analysis(symbol: str, data: pd.DataFrame):
    """
    Create performance analysis charts
    """
    returns = data['Close'].pct_change().dropna()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Returns histogram
    ax1.hist(returns, bins=50, alpha=0.7, color='#2ECC71', edgecolor='black')
    ax1.set_xlabel('Daily Returns', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Daily Returns Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    ax2.plot(data.index[1:], cumulative_returns, color='#E74C3C', linewidth=2)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Returns', fontsize=12)
    ax2.set_title('Cumulative Returns Over Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Rolling volatility
    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
    ax3.plot(data.index[30:], rolling_vol[29:], color='#9B59B6', linewidth=2)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Volatility (%)', fontsize=12)
    ax3.set_title('30-Day Rolling Volatility', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Volume vs Price
    ax4.scatter(data['Close'], data['Volume'], alpha=0.6, s=30, color='#3498DB')
    ax4.set_xlabel('Price ($)', fontsize=12)
    ax4.set_ylabel('Volume', fontsize=12)
    ax4.set_title('Volume vs Price Relationship', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../images/analysis_{symbol}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance analysis saved: ../images/analysis_{symbol}_performance.png")

def run_trading_simulation(symbol: str, capital: float = 10000, days: int = 60):
    """
    Run trading simulation and generate performance analysis
    """
    print(f"🤖 Running trading simulation for {symbol}...")
    
    # Create bot
    bot = SimpleTradingBot(symbol, capital)
    bot.confidence_threshold = 0.1  # Aggressive mode
    bot.stop_loss_pct = 0.02
    bot.take_profit_pct = 0.05
    
    # Run simulation
    bot.run_trading_session(days=days)
    
    # Create performance charts
    if bot.portfolio_value:
        create_trading_performance_charts(symbol, bot)
    
    return bot.get_performance_metrics()

def create_trading_performance_charts(symbol: str, bot: SimpleTradingBot):
    """
    Create trading performance charts
    """
    portfolio_df = pd.DataFrame(bot.portfolio_value)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio value over time
    ax1.plot(portfolio_df['date'], portfolio_df['value'], 
            color='#2ECC71', linewidth=2, label='Portfolio Value')
    ax1.axhline(y=bot.initial_capital, color='red', linestyle='--', 
               alpha=0.7, label='Initial Capital')
    ax1.set_title('Portfolio Value Evolution', fontweight='bold')
    ax1.set_ylabel('Value ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Daily returns
    if bot.daily_returns:
        ax2.plot(portfolio_df['date'][1:], bot.daily_returns, 
                color='#3498DB', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Daily Returns', fontweight='bold')
        ax2.set_ylabel('Returns (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    # Position size over time
    ax3.plot(portfolio_df['date'], portfolio_df['shares'], 
            color='#9B59B6', linewidth=2)
    ax3.set_title('Position Size Over Time', fontweight='bold')
    ax3.set_ylabel('Number of Shares', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Cash vs Position value
    ax4.plot(portfolio_df['date'], portfolio_df['cash'], 
            label='Cash', color='#E74C3C', linewidth=2)
    ax4.plot(portfolio_df['date'], portfolio_df['position_value'], 
            label='Position Value', color='#F39C12', linewidth=2)
    ax4.set_title('Cash vs Position Value', fontweight='bold')
    ax4.set_ylabel('Value ($)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis_{symbol}_trading_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Trading performance saved: analysis_{symbol}_trading_performance.png")

def main():
    """
    Generate comprehensive analysis for multiple stocks
    """
    symbols = ['AAPL', 'MSFT', 'TSLA']
    results = {}
    
    print("🚀 Starting comprehensive analysis...")
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Analyzing {symbol}...")
        print(f"{'='*60}")
        
        # Generate analysis
        analysis = generate_stock_analysis(symbol)
        
        # Run trading simulation
        metrics = run_trading_simulation(symbol, capital=10000, days=60)
        
        results[symbol] = {
            'analysis': analysis,
            'metrics': metrics
        }
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📊 SUMMARY REPORT")
    print(f"{'='*60}")
    
    for symbol, result in results.items():
        analysis = result['analysis']
        metrics = result['metrics']
        
        print(f"\n{symbol}:")
        print(f"  Current Price: ${analysis.get('current_price', 0):.2f}")
        print(f"  Total Return: {analysis.get('total_return', 0):.2f}%")
        print(f"  Volatility: {analysis.get('volatility', 0):.2f}%")
        print(f"  Trading Performance: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"  Number of Trades: {metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
    
    print(f"\n✅ Analysis complete! Check the generated PNG files for detailed visualizations.")

if __name__ == "__main__":
    main() 