"""
Script de démonstration avec simulations de trading réelles
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trading_bot_simple import SimpleTradingBot
from core.trading_bot_aggressive import AggressiveTradingBot
from core.data_collector import DataCollector
from core.technical_analysis import TechnicalAnalysis

def run_real_trading_simulation(symbol='AAPL', days=60, capital=10000, bot_type='simple'):
    """
    Exécute une vraie simulation de trading
    """
    print(f"🚀 SIMULATION DE TRADING RÉELLE - {symbol}")
    print("=" * 60)
    
    # Créer le bot approprié
    if bot_type == 'aggressive':
        bot = AggressiveTradingBot(
            symbol=symbol,
            initial_capital=capital,
            risk_per_trade=0.05,
            max_position_size=0.3
        )
        print("🤖 Bot agressif configuré")
    else:
        bot = SimpleTradingBot(
            symbol=symbol,
            initial_capital=capital,
            risk_per_trade=0.03,
            max_position_size=0.2
        )
        print("🤖 Bot simple configuré")
    
    # Préparer les données
    print("📊 Récupération des données...")
    data = bot.prepare_data(period="1y")
    
    if data is None:
        print(f"❌ Impossible de récupérer les données pour {symbol}")
        return None
    
    print(f"✅ {len(data)} enregistrements récupérés")
    
    # Exécuter la simulation
    print(f"🔄 Simulation sur {days} jours...")
    bot.run_trading_session(days=days)
    
    # Obtenir les métriques
    metrics = bot.get_performance_metrics()
    
    print(f"\n📊 RÉSULTATS DE LA SIMULATION")
    print("=" * 50)
    print(f"Rendement total: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Nombre de trades: {metrics.get('total_trades', 0)}")
    print(f"Taux de réussite: {metrics.get('win_rate', 0):.2%}")
    print(f"Gain moyen: {metrics.get('avg_win', 0):,.2f} €")
    print(f"Perte moyenne: {metrics.get('avg_loss', 0):,.2f} €")
    print(f"Capital final: {metrics.get('final_capital', capital):,.2f} €")
    
    return bot, metrics

def generate_trading_charts(bot, symbol, save_dir='../images'):
    """
    Génère des graphiques de trading avec des données réelles
    """
    print(f"📈 Génération des graphiques pour {symbol}...")
    
    # Créer le dossier si nécessaire
    os.makedirs(save_dir, exist_ok=True)
    
    # Données du bot
    data = bot.data
    trades = bot.trades
    portfolio_values = bot.portfolio_values
    
    # 1. Graphique des prix avec trades
    plt.figure(figsize=(15, 10))
    
    # Prix
    plt.subplot(2, 2, 1)
    plt.plot(data.index, data['Close'], label='Prix de clôture', linewidth=2)
    
    # Ajouter les trades
    if trades:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        if buy_trades:
            buy_dates = [t['date'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Achats', zorder=5)
        
        if sell_trades:
            sell_dates = [t['date'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Ventes', zorder=5)
    
    plt.title(f'Prix et Trades - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Prix (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Évolution du portefeuille
    plt.subplot(2, 2, 2)
    if portfolio_values:
        dates = list(portfolio_values.keys())
        values = list(portfolio_values.values())
        plt.plot(dates, values, label='Valeur du portefeuille', linewidth=2, color='blue')
        plt.axhline(y=bot.initial_capital, color='red', linestyle='--', label='Capital initial')
        plt.title(f'Évolution du Portefeuille - {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Valeur (€)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Indicateurs techniques
    plt.subplot(2, 2, 3)
    if 'RSI' in data.columns:
        plt.plot(data.index, data['RSI'], label='RSI', linewidth=2)
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        plt.title(f'RSI - {symbol}')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. Distribution des rendements
    plt.subplot(2, 2, 4)
    if trades:
        returns = []
        for trade in trades:
            if 'pnl' in trade and trade['pnl'] != 0:
                returns.append(trade['pnl'])
        
        if returns:
            plt.hist(returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', label='Break-even')
            plt.title(f'Distribution des P&L - {symbol}')
            plt.xlabel('P&L (€)')
            plt.ylabel('Fréquence')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trading_simulation_{symbol}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphique sauvegardé: trading_simulation_{symbol}.png")

def run_comparison_simulation():
    """
    Compare les performances des bots simple et agressif
    """
    print("🔄 COMPARAISON BOT SIMPLE vs AGRESSIF")
    print("=" * 60)
    
    symbols = ['AAPL', 'MSFT', 'TSLA']
    results = {}
    
    for symbol in symbols:
        print(f"\n📈 Test de {symbol}")
        print("-" * 40)
        
        # Bot simple
        print("🤖 Bot simple:")
        simple_bot, simple_metrics = run_real_trading_simulation(
            symbol=symbol, days=30, capital=5000, bot_type='simple'
        )
        
        if simple_bot:
            generate_trading_charts(simple_bot, f"{symbol}_simple")
        
        # Bot agressif
        print("\n🤖 Bot agressif:")
        aggressive_bot, aggressive_metrics = run_real_trading_simulation(
            symbol=symbol, days=30, capital=5000, bot_type='aggressive'
        )
        
        if aggressive_bot:
            generate_trading_charts(aggressive_bot, f"{symbol}_aggressive")
        
        # Comparaison
        if simple_metrics and aggressive_metrics:
            results[symbol] = {
                'simple': simple_metrics,
                'aggressive': aggressive_metrics
            }
            
            print(f"\n📊 COMPARAISON {symbol}:")
            print(f"  Simple: {simple_metrics.get('total_return_pct', 0):.2f}% ({simple_metrics.get('total_trades', 0)} trades)")
            print(f"  Agressif: {aggressive_metrics.get('total_return_pct', 0):.2f}% ({aggressive_metrics.get('total_trades', 0)} trades)")
    
    # Graphique de comparaison
    if results:
        create_comparison_chart(results)

def create_comparison_chart(results):
    """
    Crée un graphique de comparaison des performances
    """
    symbols = list(results.keys())
    simple_returns = [results[s]['simple'].get('total_return_pct', 0) for s in symbols]
    aggressive_returns = [results[s]['aggressive'].get('total_return_pct', 0) for s in symbols]
    
    x = np.arange(len(symbols))
    width = 0.35
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    bars1 = plt.bar(x - width/2, simple_returns, width, label='Bot Simple', color='lightblue')
    bars2 = plt.bar(x + width/2, aggressive_returns, width, label='Bot Agressif', color='lightcoral')
    
    plt.xlabel('Symboles')
    plt.ylabel('Rendement (%)')
    plt.title('Comparaison des Rendements')
    plt.xticks(x, symbols)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Comparaison du nombre de trades
    plt.subplot(2, 2, 2)
    simple_trades = [results[s]['simple'].get('total_trades', 0) for s in symbols]
    aggressive_trades = [results[s]['aggressive'].get('total_trades', 0) for s in symbols]
    
    bars3 = plt.bar(x - width/2, simple_trades, width, label='Bot Simple', color='lightblue')
    bars4 = plt.bar(x + width/2, aggressive_trades, width, label='Bot Agressif', color='lightcoral')
    
    plt.xlabel('Symboles')
    plt.ylabel('Nombre de Trades')
    plt.title('Comparaison du Nombre de Trades')
    plt.xticks(x, symbols)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Taux de réussite
    plt.subplot(2, 2, 3)
    simple_winrate = [results[s]['simple'].get('win_rate', 0) * 100 for s in symbols]
    aggressive_winrate = [results[s]['aggressive'].get('win_rate', 0) * 100 for s in symbols]
    
    bars5 = plt.bar(x - width/2, simple_winrate, width, label='Bot Simple', color='lightblue')
    bars6 = plt.bar(x + width/2, aggressive_winrate, width, label='Bot Agressif', color='lightcoral')
    
    plt.xlabel('Symboles')
    plt.ylabel('Taux de Réussite (%)')
    plt.title('Comparaison des Taux de Réussite')
    plt.xticks(x, symbols)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Résumé statistique
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    summary_text = "RÉSUMÉ DES PERFORMANCES\n\n"
    for symbol in symbols:
        simple = results[symbol]['simple']
        aggressive = results[symbol]['aggressive']
        
        summary_text += f"{symbol}:\n"
        summary_text += f"  Simple: {simple.get('total_return_pct', 0):.2f}% ({simple.get('total_trades', 0)} trades)\n"
        summary_text += f"  Agressif: {aggressive.get('total_return_pct', 0):.2f}% ({aggressive.get('total_trades', 0)} trades)\n\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('../images/trading_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Graphique de comparaison sauvegardé: trading_comparison.png")

def main():
    """
    Fonction principale
    """
    print("🚀 DÉMONSTRATION DE TRADING AVEC SIMULATIONS RÉELLES")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare':
            run_comparison_simulation()
        elif sys.argv[1] == 'single':
            symbol = sys.argv[2] if len(sys.argv) > 2 else 'AAPL'
            bot_type = sys.argv[3] if len(sys.argv) > 3 else 'simple'
            bot, metrics = run_real_trading_simulation(symbol, 60, 10000, bot_type)
            if bot:
                generate_trading_charts(bot, symbol)
        else:
            print("Usage: python demo_trading_real.py [compare|single] [symbol] [bot_type]")
    else:
        # Mode par défaut: comparaison complète
        run_comparison_simulation()

if __name__ == "__main__":
    main() 