"""
Script de démonstration final avec simulations de trading réelles
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def get_stock_data(symbol, period="1y"):
    """
    Récupère les données boursières via Yahoo Finance
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            print(f"❌ Aucune donnée trouvée pour {symbol}")
            return None
        
        # Nettoyer les données
        data = data.dropna()
        
        # Ajouter les indicateurs techniques
        data = add_technical_indicators(data)
        
        return data
    
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des données pour {symbol}: {str(e)}")
        return None

def add_technical_indicators(data):
    """
    Ajoute les indicateurs techniques aux données
    """
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
    
    # Moyennes mobiles
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # Bandes de Bollinger
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    return data

class TradingBot:
    """
    Bot de trading avec paramètres ajustables
    """
    
    def __init__(self, symbol, initial_capital=10000, risk_per_trade=0.05, max_position_size=0.3, aggressive=False):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.aggressive = aggressive
        
        # Paramètres de trading selon le mode
        if aggressive:
            self.confidence_threshold = 0.15  # Très permissif
            self.stop_loss_pct = 0.015  # Stop loss serré
            self.take_profit_pct = 0.04  # Take profit rapide
        else:
            self.confidence_threshold = 0.4  # Conservateur
            self.stop_loss_pct = 0.03  # Stop loss normal
            self.take_profit_pct = 0.08  # Take profit normal
        
        # État du trading
        self.position = 0
        self.shares = 0
        self.entry_price = 0
        self.trades = []
        self.portfolio_values = {}
        self.trade_days = {}  # Pour tracker les jours depuis l'achat
        
        # Données
        self.data = None
    
    def prepare_data(self, period="1y"):
        """
        Prépare les données pour le trading
        """
        self.data = get_stock_data(self.symbol, period)
        return self.data
    
    def get_technical_signals(self):
        """
        Analyse les signaux techniques
        """
        if self.data is None or len(self.data) < 50:
            return {'signal': 'HOLD', 'confidence': 0}
        
        current = self.data.iloc[-1]
        previous = self.data.iloc[-2] if len(self.data) > 1 else current
        
        signals = {}
        score = 0
        
        # RSI
        if 'RSI' in current.index:
            rsi = current['RSI']
            if self.aggressive:
                if rsi < 45:  # Plus permissif
                    signals['RSI'] = 'BUY'
                    score += 1.5
                elif rsi > 55:  # Plus permissif
                    signals['RSI'] = 'SELL'
                    score -= 1.5
                else:
                    signals['RSI'] = 'NEUTRAL'
            else:
                if rsi < 30:
                    signals['RSI'] = 'BUY'
                    score += 1
                elif rsi > 70:
                    signals['RSI'] = 'SELL'
                    score -= 1
                else:
                    signals['RSI'] = 'NEUTRAL'
        
        # MACD
        if 'MACD' in current.index and 'MACD_Signal' in current.index:
            macd = current['MACD']
            signal = current['MACD_Signal']
            macd_prev = previous['MACD'] if 'MACD' in previous.index else macd
            
            if macd > signal:
                signals['MACD'] = 'BUY'
                score += 1
            else:
                signals['MACD'] = 'SELL'
                score -= 1
        
        # Moyennes mobiles
        if 'SMA_20' in current.index:
            sma20 = current['SMA_20']
            price = current['Close']
            
            if price > sma20:
                signals['SMA'] = 'BUY'
                score += 1
            elif price < sma20:
                signals['SMA'] = 'SELL'
                score -= 1
            else:
                signals['SMA'] = 'NEUTRAL'
        
        # Momentum du prix
        price_change = (current['Close'] - previous['Close']) / previous['Close']
        if price_change > 0.005:  # Hausse de plus de 0.5%
            signals['MOMENTUM'] = 'BUY'
            score += 0.5
        elif price_change < -0.005:  # Baisse de plus de 0.5%
            signals['MOMENTUM'] = 'SELL'
            score -= 0.5
        
        # Déterminer le signal global
        if self.aggressive:
            threshold = 0.5  # Seuil bas pour agressif
        else:
            threshold = 1.0  # Seuil normal pour conservateur
        
        if score >= threshold:
            signal = 'BUY'
            confidence = min(abs(score) / 4, 1.0)
        elif score <= -threshold:
            signal = 'SELL'
            confidence = min(abs(score) / 4, 1.0)
        else:
            signal = 'HOLD'
            confidence = 0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'signals': signals,
            'score': score
        }
    
    def should_buy(self, price, current_date):
        """
        Détermine s'il faut acheter
        """
        signals = self.get_technical_signals()
        
        if signals['signal'] == 'BUY' and signals['confidence'] >= self.confidence_threshold:
            # Vérifier si on a assez de capital
            max_shares = int((self.capital * self.max_position_size) / price)
            if max_shares > 0:
                return True, max_shares
        
        # Logique supplémentaire pour agressif : premier trade
        if self.aggressive and self.position == 0 and len(self.trades) == 0:
            max_shares = int((self.capital * 0.1) / price)  # 10% du capital
            if max_shares > 0:
                return True, max_shares
        
        return False, 0
    
    def should_sell(self, price, current_date):
        """
        Détermine s'il faut vendre
        """
        if self.shares == 0:
            return False, 0
        
        signals = self.get_technical_signals()
        
        # Vendre si signal de vente
        if signals['signal'] == 'SELL' and signals['confidence'] >= self.confidence_threshold:
            return True, self.shares
        
        # Stop loss
        if price <= self.entry_price * (1 - self.stop_loss_pct):
            return True, self.shares
        
        # Take profit
        if price >= self.entry_price * (1 + self.take_profit_pct):
            return True, self.shares
        
        # Vendre après un certain temps si pas de profit (pour agressif)
        if self.aggressive and len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade['type'] == 'BUY':
                # Calculer les jours depuis l'achat
                days_held = (current_date - last_trade['date']).days
                if days_held >= 3 and price < self.entry_price * 1.01:  # Pas de profit de 1%
                    return True, self.shares
        
        return False, 0
    
    def execute_trade(self, trade_type, shares, price, date):
        """
        Exécute un trade
        """
        if trade_type == 'BUY':
            cost = shares * price
            if cost <= self.capital:
                self.capital -= cost
                self.shares += shares
                self.entry_price = price
                self.position = 1
                
                self.trades.append({
                    'date': date,
                    'type': 'BUY',
                    'shares': shares,
                    'price': price,
                    'cost': cost,
                    'confidence': self.get_technical_signals()['confidence']
                })
                
                print(f"  📈 ACHAT: {shares} actions @ {price:.2f}€ (Confiance: {self.get_technical_signals()['confidence']:.2f})")
        
        elif trade_type == 'SELL':
            revenue = shares * price
            self.capital += revenue
            self.shares -= shares
            self.position = 0
            
            # Calculer le P&L
            pnl = revenue - (shares * self.entry_price)
            
            self.trades.append({
                'date': date,
                'type': 'SELL',
                'shares': shares,
                'price': price,
                'revenue': revenue,
                'pnl': pnl,
                'confidence': self.get_technical_signals()['confidence']
            })
            
            print(f"  📉 VENTE: {shares} actions @ {price:.2f}€ (P&L: {pnl:.2f}€)")
    
    def run_trading_session(self, days=30):
        """
        Lance une session de trading
        """
        if self.data is None:
            print("❌ Aucune donnée disponible")
            return
        
        # Prendre les derniers 'days' jours
        trading_data = self.data.tail(days)
        
        print(f"🔄 Début de la session de trading sur {len(trading_data)} jours...")
        
        for i, (date, row) in enumerate(trading_data.iterrows()):
            price = row['Close']
            
            # Enregistrer la valeur du portefeuille
            portfolio_value = self.capital + (self.shares * price)
            self.portfolio_values[date] = portfolio_value
            
            # Décisions de trading
            if self.position == 0:  # Pas de position
                should_buy, shares = self.should_buy(price, date)
                if should_buy:
                    self.execute_trade('BUY', shares, price, date)
            
            else:  # Position ouverte
                should_sell, shares = self.should_sell(price, date)
                if should_sell:
                    self.execute_trade('SELL', shares, price, date)
        
        print(f"✅ Session de trading terminée")
    
    def get_performance_metrics(self):
        """
        Calcule les métriques de performance
        """
        if not self.trades:
            return {
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'final_capital': self.capital
            }
        
        # Calculer les métriques
        total_trades = len([t for t in self.trades if t['type'] == 'SELL'])
        winning_trades = len([t for t in self.trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0])
        
        wins = [t.get('pnl', 0) for t in self.trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0]
        losses = [t.get('pnl', 0) for t in self.trades if t['type'] == 'SELL' and t.get('pnl', 0) < 0]
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Calculer le rendement total
        final_value = self.capital + (self.shares * self.data['Close'].iloc[-1])
        total_return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_capital': final_value
        }

def run_trading_simulation(symbol='AAPL', days=60, capital=10000, aggressive=False):
    """
    Execute a trading simulation
    """
    mode = "AGGRESSIVE" if aggressive else "CONSERVATIVE"
    print(f"🚀 TRADING SIMULATION {mode} - {symbol}")
    print("=" * 60)
    
    # Create bot
    bot = TradingBot(
        symbol=symbol,
        initial_capital=capital,
        risk_per_trade=0.05 if aggressive else 0.03,
        max_position_size=0.3 if aggressive else 0.2,
        aggressive=aggressive
    )
    
    print(f"📊 {mode} Configuration:")
    print(f"  Capital: {capital:,.2f} €")
    print(f"  Risk per trade: {bot.risk_per_trade*100:.1f}%")
    print(f"  Max position: {bot.max_position_size*100:.1f}%")
    print(f"  Confidence threshold: {bot.confidence_threshold}")
    print(f"  Stop loss: {bot.stop_loss_pct*100:.1f}%")
    print(f"  Take profit: {bot.take_profit_pct*100:.1f}%")
    
    # Prepare data
    print("📈 Retrieving data...")
    data = bot.prepare_data(period="1y")
    
    if data is None:
        print(f"❌ Unable to retrieve data for {symbol}")
        return None, None
    
    print(f"✅ {len(data)} records retrieved")
    
    # Display current information
    current_price = data['Close'].iloc[-1]
    print(f"💰 Current price: {current_price:.2f} €")
    
    # Display technical indicators
    if 'RSI' in data.columns:
        print(f"📊 Current RSI: {data['RSI'].iloc[-1]:.2f}")
    if 'MACD' in data.columns:
        print(f"📊 Current MACD: {data['MACD'].iloc[-1]:.4f}")
    
    # Execute simulation
    print(f"🔄 Simulation over {days} days...")
    bot.run_trading_session(days=days)
    
    # Get metrics
    metrics = bot.get_performance_metrics()
    
    print(f"\n📊 SIMULATION RESULTS")
    print("=" * 50)
    print(f"Total return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Number of trades: {metrics.get('total_trades', 0)}")
    print(f"Win rate: {metrics.get('win_rate', 0):.2%}")
    print(f"Average win: {metrics.get('avg_win', 0):,.2f} €")
    print(f"Average loss: {metrics.get('avg_loss', 0):,.2f} €")
    print(f"Final capital: {metrics.get('final_capital', capital):,.2f} €")
    
    return bot, metrics

def generate_trading_charts(bot, symbol, save_dir='../images'):
    """
    Generate trading charts with real data
    """
    mode = "AGGRESSIVE" if bot.aggressive else "CONSERVATIVE"
    print(f"📈 Generating charts for {symbol} ({mode})...")
    
    # Create directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Bot data
    data = bot.data
    trades = bot.trades
    portfolio_values = bot.portfolio_values
    
    # Style configuration
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 12)
    plt.rcParams['font.size'] = 10
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Price chart with trades
    ax1 = plt.subplot(3, 2, 1)
    plt.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='blue')
    
    # Add trades
    if trades:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        if buy_trades:
            buy_dates = [t['date'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buys', zorder=5)
        
        if sell_trades:
            sell_dates = [t['date'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sells', zorder=5)
    
    plt.title(f'Price and Trades - {symbol} ({mode})', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Portfolio evolution
    ax2 = plt.subplot(3, 2, 2)
    if portfolio_values:
        dates = list(portfolio_values.keys())
        values = list(portfolio_values.values())
        plt.plot(dates, values, label='Portfolio Value', linewidth=2, color='purple')
        plt.axhline(y=bot.initial_capital, color='red', linestyle='--', label='Initial Capital')
        plt.title(f'Portfolio Evolution - {symbol}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Value (€)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Technical indicators - RSI
    ax3 = plt.subplot(3, 2, 3)
    if 'RSI' in data.columns:
        plt.plot(data.index, data['RSI'], label='RSI', linewidth=2, color='orange')
        if bot.aggressive:
            plt.axhline(y=55, color='red', linestyle='--', alpha=0.7, label='Aggressive Threshold (55)')
            plt.axhline(y=45, color='green', linestyle='--', alpha=0.7, label='Aggressive Threshold (45)')
        else:
            plt.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            plt.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        plt.title(f'RSI - {symbol}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. Technical indicators - MACD
    ax4 = plt.subplot(3, 2, 4)
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        plt.plot(data.index, data['MACD'], label='MACD', linewidth=2, color='blue')
        plt.plot(data.index, data['MACD_Signal'], label='Signal', linewidth=2, color='red')
        plt.title(f'MACD - {symbol}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Returns distribution
    ax5 = plt.subplot(3, 2, 5)
    if trades:
        returns = []
        for trade in trades:
            if 'pnl' in trade and trade['pnl'] != 0:
                returns.append(trade['pnl'])
        
        if returns:
            plt.hist(returns, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', label='Break-even')
            plt.title(f'P&L Distribution - {symbol}', fontsize=14, fontweight='bold')
            plt.xlabel('P&L (€)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 6. Performance summary
    ax6 = plt.subplot(3, 2, 6)
    plt.axis('off')
    
    metrics = bot.get_performance_metrics()
    summary_text = f"PERFORMANCE SUMMARY\n{symbol} ({mode})\n\n"
    summary_text += f"Return: {metrics.get('total_return_pct', 0):.2f}%\n"
    summary_text += f"Trades: {metrics.get('total_trades', 0)}\n"
    summary_text += f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
    summary_text += f"Avg Win: {metrics.get('avg_win', 0):,.0f} €\n"
    summary_text += f"Avg Loss: {metrics.get('avg_loss', 0):,.0f} €\n"
    summary_text += f"Final Capital: {metrics.get('final_capital', bot.initial_capital):,.0f} €\n"
    
    if trades:
        summary_text += f"\nTRADE DETAILS:\n"
        for i, trade in enumerate(trades[-5:], 1):  # Last 5 trades
            summary_text += f"{i}. {trade['type']} {trade['shares']} @ {trade['price']:.2f}€\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    suffix = "_aggressive" if bot.aggressive else "_conservative"
    plt.savefig(f'{save_dir}/trading_simulation_{symbol}{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Chart saved: trading_simulation_{symbol}{suffix}.png")

def run_comparison_simulations():
    """
    Run comparative simulations conservative vs aggressive
    """
    print("🔄 COMPARATIVE SIMULATIONS CONSERVATIVE vs AGGRESSIVE")
    print("=" * 70)
    
    symbols = ['AAPL', 'MSFT', 'TSLA']
    results = {}
    
    for symbol in symbols:
        print(f"\n📈 Testing {symbol}")
        print("-" * 50)
        
        # Conservative bot
        print("🤖 Conservative bot:")
        conservative_bot, conservative_metrics = run_trading_simulation(
            symbol=symbol, days=30, capital=5000, aggressive=False
        )
        
        if conservative_bot:
            generate_trading_charts(conservative_bot, symbol)
        
        # Aggressive bot
        print("\n🤖 Aggressive bot:")
        aggressive_bot, aggressive_metrics = run_trading_simulation(
            symbol=symbol, days=30, capital=5000, aggressive=True
        )
        
        if aggressive_bot:
            generate_trading_charts(aggressive_bot, symbol)
        
        # Comparison
        if conservative_metrics and aggressive_metrics:
            results[symbol] = {
                'conservative': conservative_metrics,
                'aggressive': aggressive_metrics
            }
            
            print(f"\n📊 COMPARISON {symbol}:")
            print(f"  Conservative: {conservative_metrics.get('total_return_pct', 0):.2f}% ({conservative_metrics.get('total_trades', 0)} trades)")
            print(f"  Aggressive: {aggressive_metrics.get('total_return_pct', 0):.2f}% ({aggressive_metrics.get('total_trades', 0)} trades)")
    
    # Comparison chart
    if results:
        create_comparison_chart(results)

def create_comparison_chart(results):
    """
    Create a performance comparison chart
    """
    print("📊 Creating comparison chart...")
    
    symbols = list(results.keys())
    conservative_returns = [results[s]['conservative'].get('total_return_pct', 0) for s in symbols]
    aggressive_returns = [results[s]['aggressive'].get('total_return_pct', 0) for s in symbols]
    conservative_trades = [results[s]['conservative'].get('total_trades', 0) for s in symbols]
    aggressive_trades = [results[s]['aggressive'].get('total_trades', 0) for s in symbols]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Returns
    x = np.arange(len(symbols))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, conservative_returns, width, label='Conservative', color='lightblue')
    bars2 = ax1.bar(x + width/2, aggressive_returns, width, label='Aggressive', color='lightcoral')
    
    ax1.set_title('Returns by Symbol', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Return (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(symbols)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Number of trades
    bars3 = ax2.bar(x - width/2, conservative_trades, width, label='Conservative', color='lightblue')
    bars4 = ax2.bar(x + width/2, aggressive_trades, width, label='Aggressive', color='lightcoral')
    
    ax2.set_title('Number of Trades', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Trades')
    ax2.set_xticks(x)
    ax2.set_xticklabels(symbols)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Win rate
    conservative_winrate = [results[s]['conservative'].get('win_rate', 0) * 100 for s in symbols]
    aggressive_winrate = [results[s]['aggressive'].get('win_rate', 0) * 100 for s in symbols]
    
    bars5 = ax3.bar(x - width/2, conservative_winrate, width, label='Conservative', color='lightblue')
    bars6 = ax3.bar(x + width/2, aggressive_winrate, width, label='Aggressive', color='lightcoral')
    
    ax3.set_title('Win Rate', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(symbols)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Summary
    ax4.axis('off')
    summary_text = "GLOBAL SUMMARY\n\n"
    for symbol in symbols:
        conservative = results[symbol]['conservative']
        aggressive = results[symbol]['aggressive']
        
        summary_text += f"{symbol}:\n"
        summary_text += f"  Conservative: {conservative.get('total_return_pct', 0):.2f}% ({conservative.get('total_trades', 0)} trades)\n"
        summary_text += f"  Aggressive: {aggressive.get('total_return_pct', 0):.2f}% ({aggressive.get('total_trades', 0)} trades)\n\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../images/trading_comparison_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comparison chart saved: trading_comparison_final.png")

def main():
    """
    Main function
    """
    print("🚀 TRADING DEMONSTRATION WITH REAL SIMULATIONS")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'single':
            symbol = sys.argv[2] if len(sys.argv) > 2 else 'AAPL'
            aggressive = sys.argv[3] == 'aggressive' if len(sys.argv) > 3 else False
            bot, metrics = run_trading_simulation(symbol, 60, 10000, aggressive)
            if bot:
                generate_trading_charts(bot, symbol)
        else:
            print("Usage: python demo_trading_final.py [single] [symbol] [conservative|aggressive]")
    else:
        # Default mode: complete comparison
        run_comparison_simulations()

if __name__ == "__main__":
    main() 