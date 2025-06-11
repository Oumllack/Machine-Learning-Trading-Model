"""
Script de démonstration du bot de trading automatique
"""

import sys
import logging
from trading_bot_simple import SimpleTradingBot

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_trading_bot():
    """
    Démonstration du bot de trading avec paramètres permissifs
    """
    print("🤖 DÉMONSTRATION DU BOT DE TRADING AUTOMATIQUE")
    print("=" * 60)
    
    # Créer le bot avec des paramètres plus permissifs
    bot = SimpleTradingBot(
        symbol='AAPL',
        initial_capital=10000,
        risk_per_trade=0.05,  # 5% de risque par trade
        max_position_size=0.2  # 20% de position max
    )
    
    # Réduire les seuils pour plus d'activité
    bot.confidence_threshold = 0.3  # Seuil de confiance plus bas
    bot.stop_loss_pct = 0.03  # Stop loss plus serré
    bot.take_profit_pct = 0.08  # Take profit plus rapide
    
    print(f"📊 Configuration du bot:")
    print(f"  Symbole: {bot.symbol}")
    print(f"  Capital: {bot.initial_capital:,.2f} €")
    print(f"  Risque par trade: {bot.risk_per_trade*100:.1f}%")
    print(f"  Position max: {bot.max_position_size*100:.1f}%")
    print(f"  Seuil confiance: {bot.confidence_threshold}")
    print(f"  Stop loss: {bot.stop_loss_pct*100:.1f}%")
    print(f"  Take profit: {bot.take_profit_pct*100:.1f}%")
    
    # Préparer les données
    print(f"\n📈 Préparation des données...")
    data = bot.prepare_data(period="1y")
    
    if data is None:
        print("❌ Impossible de récupérer les données")
        return
    
    print(f"✅ Données récupérées: {len(data)} enregistrements")
    
    # Afficher les informations actuelles
    current_price = data['Close'].iloc[-1]
    print(f"💰 Prix actuel: {current_price:.2f} €")
    
    # Afficher les indicateurs techniques
    print(f"\n📊 Indicateurs techniques actuels:")
    print(f"  RSI: {data['RSI'].iloc[-1]:.2f}")
    print(f"  MACD: {data['MACD'].iloc[-1]:.4f}")
    print(f"  SMA 20: {data['SMA_20'].iloc[-1]:.2f}")
    print(f"  SMA 50: {data['SMA_50'].iloc[-1]:.2f}")
    print(f"  SMA 200: {data['SMA_200'].iloc[-1]:.2f}")
    
    # Obtenir les signaux
    technical_signals = bot.get_technical_signals()
    print(f"\n🎯 Signaux techniques:")
    print(f"  Signal global: {technical_signals.get('overall_signal', 'N/A')}")
    print(f"  Score technique: {technical_signals.get('technical_score', 0):.3f}")
    
    # Afficher les signaux détaillés
    signals = technical_signals.get('signals', {})
    for signal_type, signal_value in signals.items():
        print(f"    {signal_type}: {signal_value}")
    
    # Obtenir la prédiction
    prediction = bot.get_simple_prediction(days_ahead=5)
    if prediction:
        print(f"\n🧠 Prédiction technique:")
        print(f"  Tendance: {prediction['trend']}")
        print(f"  Confiance: {prediction['confidence']:.2f}")
        print(f"  Momentum: {prediction['momentum']:.2f}%")
        print(f"  Volatilité: {prediction['volatility']:.4f}")
        print(f"  Prédictions (5 jours): {prediction['prediction'][:3]}...")
    
    # Simuler le trading sur une période plus longue
    print(f"\n🔄 Simulation de trading (60 jours)...")
    bot.run_trading_session(days=60)
    
    # Afficher le rapport de performance
    print(f"\n" + "="*60)
    bot.print_performance_report()
    
    # Afficher les détails des trades
    if bot.trades:
        print(f"\n📋 Détail des trades:")
        for i, trade in enumerate(bot.trades, 1):
            print(f"  Trade {i}:")
            print(f"    Type: {trade['type']}")
            print(f"    Date: {trade['date']}")
            print(f"    Prix: {trade['price']:.2f} €")
            print(f"    Actions: {trade['shares']}")
            print(f"    Confiance: {trade['confidence']:.2f}")
            if 'pnl' in trade:
                print(f"    P&L: {trade['pnl']:.2f} €")
            print()
    
    # Sauvegarder les résultats
    bot.save_trading_log("demo_trading_results")
    print(f"💾 Résultats sauvegardés dans demo_trading_results.csv")
    
    print(f"\n✅ Démonstration terminée!")

def demo_multiple_symbols():
    """
    Démonstration avec plusieurs symboles
    """
    print("🔄 DÉMONSTRATION AVEC PLUSIEURS SYMBOLES")
    print("=" * 60)
    
    symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL']
    results = {}
    
    for symbol in symbols:
        print(f"\n📈 Test de {symbol}")
        print("-" * 30)
        
        try:
            bot = SimpleTradingBot(
                symbol=symbol,
                initial_capital=5000,
                risk_per_trade=0.05,
                max_position_size=0.2
            )
            
            # Paramètres permissifs
            bot.confidence_threshold = 0.3
            bot.stop_loss_pct = 0.03
            bot.take_profit_pct = 0.08
            
            # Préparer et trader
            data = bot.prepare_data(period="6m")
            if data is None:
                print(f"❌ Impossible de récupérer les données pour {symbol}")
                continue
            
            print(f"✅ Données: {len(data)} enregistrements")
            
            # Trading
            bot.run_trading_session(days=30)
            
            # Métriques
            metrics = bot.get_performance_metrics()
            results[symbol] = metrics
            
            print(f"💰 Rendement: {metrics.get('total_return_pct', 0):.2f}%")
            print(f"📊 Trades: {metrics.get('total_trades', 0)}")
            print(f"🎯 Taux de réussite: {metrics.get('win_rate', 0):.2%}")
            
        except Exception as e:
            print(f"❌ Erreur pour {symbol}: {str(e)}")
            continue
    
    # Résumé
    print(f"\n📊 RÉSUMÉ DES RÉSULTATS")
    print("=" * 60)
    for symbol, metrics in results.items():
        print(f"{symbol}: {metrics.get('total_return_pct', 0):.2f}% ({metrics.get('total_trades', 0)} trades)")

def main():
    """
    Fonction principale
    """
    if len(sys.argv) > 1 and sys.argv[1] == 'multi':
        demo_multiple_symbols()
    else:
        demo_trading_bot()

if __name__ == "__main__":
    main() 