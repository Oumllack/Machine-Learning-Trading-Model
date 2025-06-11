"""
Script de démonstration simple avec simulations de trading réelles
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports directs pour éviter les problèmes de dépendances
from core.data_collector import DataCollector
from core.technical_analysis import TechnicalAnalysis
from core.trading_bot_simple import SimpleTradingBot

def run_trading_simulation(symbol='AAPL', days=60, capital=10000):
    """
    Exécute une simulation de trading réelle
    """
    print(f"🚀 SIMULATION DE TRADING - {symbol}")
    print("=" * 60)
    
    # Créer le bot
    bot = SimpleTradingBot(
        symbol=symbol,
        initial_capital=capital,
        risk_per_trade=0.03,
        max_position_size=0.2
    )
    
    # Paramètres plus permissifs pour plus d'activité
    bot.confidence_threshold = 0.4
    bot.stop_loss_pct = 0.02
    bot.take_profit_pct = 0.06
    
    print(f"📊 Configuration:")
    print(f"  Capital: {capital:,.2f} €")
    print(f"  Risque par trade: {bot.risk_per_trade*100:.1f}%")
    print(f"  Position max: {bot.max_position_size*100:.1f}%")
    print(f"  Seuil confiance: {bot.confidence_threshold}")
    
    # Préparer les données
    print("📈 Récupération des données...")
    data = bot.prepare_data(period="1y")
    
    if data is None:
        print(f"❌ Impossible de récupérer les données pour {symbol}")
        return None, None
    
    print(f"✅ {len(data)} enregistrements récupérés")
    
    # Afficher les informations actuelles
    current_price = data['Close'].iloc[-1]
    print(f"💰 Prix actuel: {current_price:.2f} €")
    
    # Afficher les indicateurs techniques
    if 'RSI' in data.columns:
        print(f"📊 RSI actuel: {data['RSI'].iloc[-1]:.2f}")
    if 'MACD' in data.columns:
        print(f"📊 MACD actuel: {data['MACD'].iloc[-1]:.4f}")
    
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
    
    # Configuration du style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 12)
    plt.rcParams['font.size'] = 10
    
    # Créer la figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Graphique des prix avec trades
    ax1 = plt.subplot(3, 2, 1)
    plt.plot(data.index, data['Close'], label='Prix de clôture', linewidth=2, color='blue')
    
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
    
    plt.title(f'Prix et Trades - {symbol}', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Prix (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Évolution du portefeuille
    ax2 = plt.subplot(3, 2, 2)
    if portfolio_values:
        dates = list(portfolio_values.keys())
        values = list(portfolio_values.values())
        plt.plot(dates, values, label='Valeur du portefeuille', linewidth=2, color='purple')
        plt.axhline(y=bot.initial_capital, color='red', linestyle='--', label='Capital initial')
        plt.title(f'Évolution du Portefeuille - {symbol}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Valeur (€)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. Indicateurs techniques - RSI
    ax3 = plt.subplot(3, 2, 3)
    if 'RSI' in data.columns:
        plt.plot(data.index, data['RSI'], label='RSI', linewidth=2, color='orange')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Surachat (70)')
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Survente (30)')
        plt.title(f'RSI - {symbol}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. Indicateurs techniques - MACD
    ax4 = plt.subplot(3, 2, 4)
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        plt.plot(data.index, data['MACD'], label='MACD', linewidth=2, color='blue')
        plt.plot(data.index, data['MACD_Signal'], label='Signal', linewidth=2, color='red')
        plt.title(f'MACD - {symbol}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Distribution des rendements
    ax5 = plt.subplot(3, 2, 5)
    if trades:
        returns = []
        for trade in trades:
            if 'pnl' in trade and trade['pnl'] != 0:
                returns.append(trade['pnl'])
        
        if returns:
            plt.hist(returns, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', label='Break-even')
            plt.title(f'Distribution des P&L - {symbol}', fontsize=14, fontweight='bold')
            plt.xlabel('P&L (€)')
            plt.ylabel('Fréquence')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 6. Résumé des performances
    ax6 = plt.subplot(3, 2, 6)
    plt.axis('off')
    
    metrics = bot.get_performance_metrics()
    summary_text = f"RÉSUMÉ DES PERFORMANCES\n{symbol}\n\n"
    summary_text += f"Rendement: {metrics.get('total_return_pct', 0):.2f}%\n"
    summary_text += f"Trades: {metrics.get('total_trades', 0)}\n"
    summary_text += f"Taux de réussite: {metrics.get('win_rate', 0):.2%}\n"
    summary_text += f"Gain moyen: {metrics.get('avg_win', 0):,.0f} €\n"
    summary_text += f"Perte moyenne: {metrics.get('avg_loss', 0):,.0f} €\n"
    summary_text += f"Capital final: {metrics.get('final_capital', bot.initial_capital):,.0f} €\n"
    
    if trades:
        summary_text += f"\nDÉTAIL DES TRADES:\n"
        for i, trade in enumerate(trades[-5:], 1):  # Derniers 5 trades
            summary_text += f"{i}. {trade['type']} {trade['shares']} @ {trade['price']:.2f}€\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trading_simulation_{symbol}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphique sauvegardé: trading_simulation_{symbol}.png")

def run_multiple_simulations():
    """
    Lance des simulations sur plusieurs symboles
    """
    print("🔄 SIMULATIONS MULTIPLES")
    print("=" * 60)
    
    symbols = ['AAPL', 'MSFT', 'TSLA']
    results = {}
    
    for symbol in symbols:
        print(f"\n📈 Simulation pour {symbol}")
        print("-" * 40)
        
        bot, metrics = run_trading_simulation(symbol, 30, 5000)
        
        if bot and metrics:
            results[symbol] = metrics
            generate_trading_charts(bot, symbol)
            
            print(f"✅ Simulation terminée pour {symbol}")
        else:
            print(f"❌ Échec de la simulation pour {symbol}")
    
    # Créer un graphique de comparaison
    if results:
        create_comparison_chart(results)

def create_comparison_chart(results):
    """
    Crée un graphique de comparaison des performances
    """
    print("📊 Création du graphique de comparaison...")
    
    symbols = list(results.keys())
    returns = [results[s].get('total_return_pct', 0) for s in symbols]
    trades = [results[s].get('total_trades', 0) for s in symbols]
    win_rates = [results[s].get('win_rate', 0) * 100 for s in symbols]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Rendements
    bars1 = ax1.bar(symbols, returns, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax1.set_title('Rendements par Symbole', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Rendement (%)')
    ax1.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Nombre de trades
    bars2 = ax2.bar(symbols, trades, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax2.set_title('Nombre de Trades', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Nombre de Trades')
    ax2.grid(True, alpha=0.3)
    
    # Taux de réussite
    bars3 = ax3.bar(symbols, win_rates, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax3.set_title('Taux de Réussite', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Taux de Réussite (%)')
    ax3.grid(True, alpha=0.3)
    
    # Résumé
    ax4.axis('off')
    summary_text = "RÉSUMÉ GLOBAL\n\n"
    for symbol in symbols:
        metrics = results[symbol]
        summary_text += f"{symbol}:\n"
        summary_text += f"  Rendement: {metrics.get('total_return_pct', 0):.2f}%\n"
        summary_text += f"  Trades: {metrics.get('total_trades', 0)}\n"
        summary_text += f"  Taux de réussite: {metrics.get('win_rate', 0):.2%}\n\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
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
        if sys.argv[1] == 'single':
            symbol = sys.argv[2] if len(sys.argv) > 2 else 'AAPL'
            bot, metrics = run_trading_simulation(symbol, 60, 10000)
            if bot:
                generate_trading_charts(bot, symbol)
        else:
            print("Usage: python demo_trading_simple.py [single] [symbol]")
    else:
        # Mode par défaut: simulations multiples
        run_multiple_simulations()

if __name__ == "__main__":
    main() 