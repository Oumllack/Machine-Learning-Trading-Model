"""
Script de test pour le bot de trading automatique
"""

import sys
import logging
from trading_bot import TradingBot

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_trading_bot():
    """
    Teste le bot de trading avec différentes actions
    """
    print("🤖 TEST DU BOT DE TRADING AUTOMATIQUE")
    print("=" * 50)
    
    # Liste des actions à tester
    symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL']
    
    for symbol in symbols:
        print(f"\n📈 Test du bot pour {symbol}")
        print("-" * 30)
        
        try:
            # Créer le bot de trading
            bot = TradingBot(
                symbol=symbol,
                initial_capital=10000,
                risk_per_trade=0.02,
                max_position_size=0.1
            )
            
            # Préparer les données
            data = bot.prepare_data(period="1y")
            if data is None:
                print(f"❌ Impossible de récupérer les données pour {symbol}")
                continue
            
            print(f"✅ Données récupérées: {len(data)} enregistrements")
            
            # Obtenir les signaux techniques
            technical_signals = bot.get_technical_signals()
            print(f"📊 Signaux techniques: {technical_signals.get('overall_signal', 'N/A')}")
            
            # Obtenir la prédiction LSTM
            prediction = bot.get_lstm_prediction(days_ahead=5)
            if prediction:
                print(f"🧠 Prédiction LSTM: {prediction['trend']} (confiance: {prediction['confidence']:.2f})")
            else:
                print("⚠️  Prédiction LSTM non disponible")
            
            # Simuler quelques jours de trading
            print("\n🔄 Simulation de trading...")
            bot.run_trading_session(days=10)
            
            # Afficher les métriques de performance
            metrics = bot.get_performance_metrics()
            if metrics:
                print(f"💰 Rendement: {metrics.get('total_return_pct', 0):.2f}%")
                print(f"📊 Trades: {metrics.get('total_trades', 0)}")
                print(f"🎯 Taux de réussite: {metrics.get('win_rate', 0):.2%}")
            
            # Sauvegarder les résultats
            bot.save_trading_log()
            
        except Exception as e:
            print(f"❌ Erreur lors du test de {symbol}: {str(e)}")
            continue
    
    print("\n" + "=" * 50)
    print("✅ Test du bot de trading terminé")

def test_single_symbol(symbol: str = 'AAPL'):
    """
    Teste le bot avec un seul symbole en détail
    """
    print(f"\n🎯 TEST DÉTAILLÉ POUR {symbol}")
    print("=" * 50)
    
    try:
        # Créer le bot
        bot = TradingBot(symbol, initial_capital=10000)
        
        # Préparer les données
        data = bot.prepare_data()
        if data is None:
            print("❌ Impossible de récupérer les données")
            return
        
        print(f"✅ Données préparées: {len(data)} enregistrements")
        
        # Afficher les dernières données
        latest_data = data.tail(5)
        print(f"\n📈 Dernières données:")
        print(latest_data[['Close', 'Volume', 'RSI', 'MACD']].to_string())
        
        # Obtenir les signaux
        technical_signals = bot.get_technical_signals()
        print(f"\n📊 Signaux techniques:")
        for signal_type, signal_value in technical_signals.get('signals', {}).items():
            print(f"  {signal_type}: {signal_value}")
        
        # Prédiction LSTM
        prediction = bot.get_lstm_prediction()
        if prediction:
            print(f"\n🧠 Prédiction LSTM:")
            print(f"  Tendance: {prediction['trend']}")
            print(f"  Confiance: {prediction['confidence']:.2f}")
            print(f"  Prédiction: {prediction['prediction'][:3]}...")  # Afficher les 3 premières valeurs
        
        # Simuler le trading
        print(f"\n🔄 Simulation de trading (30 jours)...")
        bot.run_trading_session(days=30)
        
        # Rapport de performance
        bot.print_performance_report()
        
        # Sauvegarder
        bot.save_trading_log()
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """
    Fonction principale
    """
    print("🚀 DÉMARRAGE DU TEST DU BOT DE TRADING")
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
        test_single_symbol(symbol)
    else:
        # Test rapide avec plusieurs symboles
        test_trading_bot()
        
        # Test détaillé avec Apple
        test_single_symbol('AAPL')

if __name__ == "__main__":
    main() 