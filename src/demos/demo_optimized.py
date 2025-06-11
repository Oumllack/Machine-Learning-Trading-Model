"""
Demo script with optimized parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from trading_bot_simple import SimpleTradingBot
from data_collector import DataCollector
from technical_analysis import TechnicalAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_optimized_demo():
    """
    Run demo with optimized parameters
    """
    print("🚀 Advanced Trading System - Optimized Demo")
    print("="*60)
    
    # Optimized parameters based on analysis
    optimized_params = {
        'confidence_threshold': 0.4,  # More aggressive
        'stop_loss_pct': 0.03,        # Tighter stop loss
        'take_profit_pct': 0.10,      # Realistic take profit
        'trailing_stop_pct': 0.02     # Trailing stop
    }
    
    # Test stocks
    symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN']
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol} with optimized parameters...")
        print("-" * 40)
        
        # Create bot with optimized parameters
        bot = SimpleTradingBot(symbol, 10000)
        
        # Apply optimized parameters
        bot.confidence_threshold = optimized_params['confidence_threshold']
        bot.stop_loss_pct = optimized_params['stop_loss_pct']
        bot.take_profit_pct = optimized_params['take_profit_pct']
        
        # Run trading session
        bot.run_trading_session(days=90)  # Longer period for better results
        
        # Get performance metrics
        metrics = bot.get_performance_metrics()
        
        # Calculate additional metrics
        if bot.daily_returns:
            returns = np.array(bot.daily_returns)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = np.min(returns) if len(returns) > 0 else 0
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Calculate current portfolio value
        current_value = bot.cash + (bot.shares * bot.current_price) if hasattr(bot, 'current_price') else bot.cash
        
        results[symbol] = {
            'metrics': metrics,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_value': current_value,
            'bot': bot
        }
        
        # Print results
        print(f"Initial Capital: ${bot.initial_capital:,.2f}")
        print(f"Final Value: ${current_value:,.2f}")
        print(f"Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"Number of Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        
        # Create performance chart
        create_performance_chart(symbol, bot, optimized_params)
    
    # Portfolio analysis
    print(f"\n📈 PORTFOLIO ANALYSIS")
    print("="*60)
    
    portfolio_value = 0
    total_trades = 0
    total_wins = 0
    
    for symbol, result in results.items():
        portfolio_value += result['current_value']
        total_trades += result['metrics']['total_trades']
        total_wins += result['metrics']['total_trades'] * result['metrics']['win_rate']
    
    initial_capital = len(symbols) * 10000
    portfolio_return = (portfolio_value / initial_capital - 1) * 100
    overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
    
    print(f"Initial Portfolio Value: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Portfolio Return: {portfolio_return:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Overall Win Rate: {overall_win_rate:.2%}")
    
    # Create portfolio comparison chart
    create_portfolio_comparison(results, optimized_params)
    
    return results

def create_performance_chart(symbol: str, bot: SimpleTradingBot, params: dict):
    """
    Create performance chart for individual stock
    """
    if not bot.portfolio_value:
        return
    
    portfolio_df = pd.DataFrame(bot.portfolio_value)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio value over time
    ax1.plot(portfolio_df['date'], portfolio_df['value'], 
            color='#2ECC71', linewidth=2, label='Portfolio Value')
    ax1.axhline(y=bot.initial_capital, color='red', linestyle='--', 
               alpha=0.7, label='Initial Capital')
    ax1.set_title(f'{symbol} - Portfolio Value Evolution (Optimized)', fontweight='bold')
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
    
    # Add parameter info
    fig.suptitle(f'{symbol} - Optimized Parameters: Confidence={params["confidence_threshold"]}, '
                f'Stop Loss={params["stop_loss_pct"]*100}%, Take Profit={params["take_profit_pct"]*100}%', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'optimized_{symbol}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance chart saved: optimized_{symbol}_performance.png")

def create_portfolio_comparison(results: dict, params: dict):
    """
    Create portfolio comparison chart
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio returns comparison
    symbols = list(results.keys())
    returns = [results[s]['metrics']['total_return_pct'] for s in symbols]
    
    bars = ax1.bar(symbols, returns, color=['#2ECC71', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12'])
    ax1.set_title('Portfolio Returns by Stock', fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, return_val in zip(bars, returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{return_val:.1f}%', ha='center', va='bottom')
    
    # Sharpe ratios
    sharpe_ratios = [results[s]['sharpe_ratio'] for s in symbols]
    ax2.bar(symbols, sharpe_ratios, color=['#2ECC71', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12'])
    ax2.set_title('Sharpe Ratios by Stock', fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Number of trades
    trades = [results[s]['metrics']['total_trades'] for s in symbols]
    ax3.bar(symbols, trades, color=['#2ECC71', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12'])
    ax3.set_title('Number of Trades by Stock', fontweight='bold')
    ax3.set_ylabel('Number of Trades', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Win rates
    win_rates = [results[s]['metrics']['win_rate'] * 100 for s in symbols]
    ax4.bar(symbols, win_rates, color=['#2ECC71', '#3498DB', '#9B59B6', '#E74C3C', '#F39C12'])
    ax4.set_title('Win Rates by Stock', fontweight='bold')
    ax4.set_ylabel('Win Rate (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add parameter info
    fig.suptitle('Portfolio Performance Summary - Optimized Parameters', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('optimized_portfolio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Portfolio comparison saved: optimized_portfolio_comparison.png")

def compare_with_baseline():
    """
    Compare optimized parameters with baseline
    """
    print(f"\n📊 COMPARISON: Optimized vs Baseline Parameters")
    print("="*60)
    
    symbol = 'MSFT'  # Use Microsoft as example
    
    # Baseline parameters (conservative)
    baseline_params = {
        'confidence_threshold': 0.6,
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.15
    }
    
    # Optimized parameters
    optimized_params = {
        'confidence_threshold': 0.4,
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.10
    }
    
    # Test baseline
    print("Testing baseline parameters...")
    baseline_bot = SimpleTradingBot(symbol, 10000)
    baseline_bot.confidence_threshold = baseline_params['confidence_threshold']
    baseline_bot.stop_loss_pct = baseline_params['stop_loss_pct']
    baseline_bot.take_profit_pct = baseline_params['take_profit_pct']
    baseline_bot.run_trading_session(days=90)
    baseline_metrics = baseline_bot.get_performance_metrics()
    
    # Test optimized
    print("Testing optimized parameters...")
    optimized_bot = SimpleTradingBot(symbol, 10000)
    optimized_bot.confidence_threshold = optimized_params['confidence_threshold']
    optimized_bot.stop_loss_pct = optimized_params['stop_loss_pct']
    optimized_bot.take_profit_pct = optimized_params['take_profit_pct']
    optimized_bot.run_trading_session(days=90)
    optimized_metrics = optimized_bot.get_performance_metrics()
    
    # Calculate current values
    baseline_value = baseline_bot.cash + (baseline_bot.shares * baseline_bot.current_price) if hasattr(baseline_bot, 'current_price') else baseline_bot.cash
    optimized_value = optimized_bot.cash + (optimized_bot.shares * optimized_bot.current_price) if hasattr(optimized_bot, 'current_price') else optimized_bot.cash
    
    # Print comparison
    print(f"\nBaseline Parameters:")
    print(f"  Confidence Threshold: {baseline_params['confidence_threshold']}")
    print(f"  Stop Loss: {baseline_params['stop_loss_pct']*100}%")
    print(f"  Take Profit: {baseline_params['take_profit_pct']*100}%")
    print(f"  Final Value: ${baseline_value:,.2f}")
    print(f"  Total Return: {baseline_metrics['total_return_pct']:.2f}%")
    print(f"  Number of Trades: {baseline_metrics['total_trades']}")
    print(f"  Win Rate: {baseline_metrics['win_rate']:.2%}")
    
    print(f"\nOptimized Parameters:")
    print(f"  Confidence Threshold: {optimized_params['confidence_threshold']}")
    print(f"  Stop Loss: {optimized_params['stop_loss_pct']*100}%")
    print(f"  Take Profit: {optimized_params['take_profit_pct']*100}%")
    print(f"  Final Value: ${optimized_value:,.2f}")
    print(f"  Total Return: {optimized_metrics['total_return_pct']:.2f}%")
    print(f"  Number of Trades: {optimized_metrics['total_trades']}")
    print(f"  Win Rate: {optimized_metrics['win_rate']:.2%}")
    
    # Calculate improvement
    return_improvement = optimized_metrics['total_return_pct'] - baseline_metrics['total_return_pct']
    trade_improvement = optimized_metrics['total_trades'] - baseline_metrics['total_trades']
    
    print(f"\nImprovements:")
    print(f"  Return Improvement: {return_improvement:+.2f}%")
    print(f"  Trade Activity: {trade_improvement:+d} trades")
    
    # Create comparison chart
    create_comparison_chart(symbol, baseline_bot, optimized_bot, 
                          baseline_params, optimized_params)

def create_comparison_chart(symbol: str, baseline_bot: SimpleTradingBot, 
                          optimized_bot: SimpleTradingBot, baseline_params: dict, 
                          optimized_params: dict):
    """
    Create comparison chart between baseline and optimized parameters
    """
    if not baseline_bot.portfolio_value or not optimized_bot.portfolio_value:
        return
    
    baseline_df = pd.DataFrame(baseline_bot.portfolio_value)
    optimized_df = pd.DataFrame(optimized_bot.portfolio_value)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Portfolio value comparison
    ax1.plot(baseline_df['date'], baseline_df['value'], 
            label='Baseline', color='#E74C3C', linewidth=2)
    ax1.plot(optimized_df['date'], optimized_df['value'], 
            label='Optimized', color='#2ECC71', linewidth=2)
    ax1.axhline(y=baseline_bot.initial_capital, color='black', linestyle='--', 
               alpha=0.7, label='Initial Capital')
    ax1.set_title(f'{symbol} - Portfolio Value Comparison', fontweight='bold')
    ax1.set_ylabel('Value ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance metrics comparison
    metrics = ['Total Return (%)', 'Number of Trades', 'Win Rate (%)']
    baseline_values = [
        baseline_bot.get_performance_metrics()['total_return_pct'],
        baseline_bot.get_performance_metrics()['total_trades'],
        baseline_bot.get_performance_metrics()['win_rate'] * 100
    ]
    optimized_values = [
        optimized_bot.get_performance_metrics()['total_return_pct'],
        optimized_bot.get_performance_metrics()['total_trades'],
        optimized_bot.get_performance_metrics()['win_rate'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax2.bar(x - width/2, baseline_values, width, label='Baseline', color='#E74C3C', alpha=0.7)
    ax2.bar(x + width/2, optimized_values, width, label='Optimized', color='#2ECC71', alpha=0.7)
    
    ax2.set_title('Performance Metrics Comparison', fontweight='bold')
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{symbol}_baseline_vs_optimized.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison chart saved: comparison_{symbol}_baseline_vs_optimized.png")

def main():
    """
    Main demo function
    """
    print("🎯 Advanced Trading System - Optimized Demo")
    print("This demo shows the system with optimized parameters for better performance")
    print("="*80)
    
    # Run optimized demo
    results = run_optimized_demo()
    
    # Compare with baseline
    compare_with_baseline()
    
    print(f"\n✅ Demo completed! Check the generated PNG files for detailed analysis.")
    print(f"Key improvements with optimized parameters:")
    print(f"  - More aggressive confidence threshold (0.4 vs 0.6)")
    print(f"  - Tighter stop loss (3% vs 5%)")
    print(f"  - Realistic take profit (10% vs 15%)")
    print(f"  - Better risk-reward ratio")
    print(f"  - Increased trading activity")

if __name__ == "__main__":
    main() 