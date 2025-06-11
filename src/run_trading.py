"""
Script principal pour exécuter le bot de trading automatique
"""

import sys
import argparse
import logging
from datetime import datetime
from trading_bot import TradingBot

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'trading_log_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Fonction principale avec interface en ligne de commande
    """
    parser = argparse.ArgumentParser(description='Bot de trading automatique avec IA')
    parser.add_argument('--symbol', '-s', type=str, default='AAPL', 
                       help='Symbole de l\'action à trader (défaut: AAPL)')
    parser.add_argument('--capital', '-c', type=float, default=10000,
                       help='Capital initial en euros (défaut: 10000)')
    parser.add_argument('--days', '-d', type=int, default=30,
                       help='Nombre de jours de simulation (défaut: 30)')
    parser.add_argument('--risk', '-r', type=float, default=0.02,
                       help='Risque par trade en pourcentage (défaut: 0.02)')
    parser.add_argument('--max-position', '-m', type=float, default=0.1,
                       help='Taille maximale de position en pourcentage (défaut: 0.1)')
    parser.add_argument('--mode', type=str, choices=['simulation', 'live'], default='simulation',
                       help='Mode de trading (défaut: simulation)')
    
    args = parser.parse_args()
    
    print("🤖 BOT DE TRADING AUTOMATIQUE AVEC IA")
    print("=" * 50)
    print(f"Symbole: {args.symbol}")
    print(f"Capital initial: {args.capital:,.2f} €")
    print(f"Mode: {args.mode}")
    print(f"Durée: {args.days} jours")
    print("=" * 50)
    
    try:
        # Créer le bot de trading
        bot = TradingBot(
            symbol=args.symbol,
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            max_position_size=args.max_position
        )
        
        # Préparer les données
        print("\n📊 Préparation des données...")
        data = bot.prepare_data(period="2y")
        
        if data is None:
            print("❌ Impossible de récupérer les données")
            return
        
        print(f"✅ Données récupérées: {len(data)} enregistrements")
        
        # Afficher les informations actuelles
        current_price = data['Close'].iloc[-1]
        print(f"💰 Prix actuel: {current_price:.2f} €")
        
        # Obtenir les signaux techniques
        print("\n📈 Analyse technique...")
        technical_signals = bot.get_technical_signals()
        print(f"Signal global: {technical_signals.get('overall_signal', 'N/A')}")
        print(f"Score technique: {technical_signals.get('technical_score', 0):.3f}")
        
        # Obtenir la prédiction LSTM
        print("\n🧠 Prédiction IA...")
        prediction = bot.get_lstm_prediction(days_ahead=5)
        
        if prediction:
            print(f"Tendance: {prediction['trend']}")
            print(f"Confiance: {prediction['confidence']:.2f}")
            print(f"Prédictions (5 jours): {prediction['prediction'][:3]}...")
        else:
            print("⚠️  Prédiction non disponible")
        
        # Confirmation pour continuer
        if args.mode == 'live':
            response = input("\n⚠️  ATTENTION: Mode LIVE activé. Continuer? (oui/non): ")
            if response.lower() not in ['oui', 'yes', 'o', 'y']:
                print("❌ Opération annulée")
                return
        
        # Exécuter la session de trading
        print(f"\n🔄 Démarrage de la session de trading ({args.days} jours)...")
        bot.run_trading_session(days=args.days)
        
        # Afficher le rapport de performance
        print("\n" + "=" * 50)
        bot.print_performance_report()
        
        # Sauvegarder les résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_results_{args.symbol}_{timestamp}"
        bot.save_trading_log(filename)
        
        print(f"\n💾 Résultats sauvegardés: {filename}")
        
        # Afficher les derniers trades
        if bot.trades:
            print(f"\n📋 Derniers trades:")
            for trade in bot.trades[-5:]:  # Afficher les 5 derniers trades
                print(f"  {trade['type']}: {trade['shares']} actions à {trade['price']:.2f} €")
                if 'pnl' in trade:
                    print(f"    P&L: {trade['pnl']:.2f} €")
        
        print("\n✅ Session de trading terminée avec succès!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Session interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        logger.error(f"Erreur lors de l'exécution: {str(e)}", exc_info=True)

def interactive_mode():
    """
    Mode interactif pour configurer le bot
    """
    print("🎮 MODE INTERACTIF - CONFIGURATION DU BOT")
    print("=" * 50)
    
    # Demander les paramètres
    symbol = input("Symbole de l'action (ex: AAPL, MSFT, TSLA): ").strip().upper()
    if not symbol:
        symbol = 'AAPL'
    
    try:
        capital = float(input("Capital initial en euros (défaut: 10000): ") or "10000")
    except ValueError:
        capital = 10000
    
    try:
        days = int(input("Nombre de jours de simulation (défaut: 30): ") or "30")
    except ValueError:
        days = 30
    
    try:
        risk = float(input("Risque par trade en % (défaut: 2): ") or "2") / 100
    except ValueError:
        risk = 0.02
    
    mode = input("Mode (simulation/live) [défaut: simulation]: ").strip().lower()
    if mode not in ['live', 'simulation']:
        mode = 'simulation'
    
    print(f"\n✅ Configuration:")
    print(f"  Symbole: {symbol}")
    print(f"  Capital: {capital:,.2f} €")
    print(f"  Durée: {days} jours")
    print(f"  Risque: {risk*100:.1f}%")
    print(f"  Mode: {mode}")
    
    # Créer les arguments et exécuter
    class Args:
        def __init__(self):
            self.symbol = symbol
            self.capital = capital
            self.days = days
            self.risk = risk
            self.max_position = 0.1
            self.mode = mode
    
    args = Args()
    
    # Exécuter le trading
    try:
        bot = TradingBot(
            symbol=args.symbol,
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            max_position_size=args.max_position
        )
        
        print("\n📊 Préparation des données...")
        data = bot.prepare_data()
        
        if data is None:
            print("❌ Impossible de récupérer les données")
            return
        
        print(f"✅ Données récupérées: {len(data)} enregistrements")
        
        # Afficher les informations
        current_price = data['Close'].iloc[-1]
        print(f"💰 Prix actuel: {current_price:.2f} €")
        
        # Signaux techniques
        technical_signals = bot.get_technical_signals()
        print(f"📊 Signal technique: {technical_signals.get('overall_signal', 'N/A')}")
        
        # Prédiction IA
        prediction = bot.get_lstm_prediction()
        if prediction:
            print(f"🧠 Prédiction IA: {prediction['trend']} (confiance: {prediction['confidence']:.2f})")
        
        # Confirmation
        if args.mode == 'live':
            response = input("\n⚠️  Mode LIVE. Continuer? (oui/non): ")
            if response.lower() not in ['oui', 'yes', 'o', 'y']:
                print("❌ Annulé")
                return
        
        # Exécution
        print(f"\n🔄 Démarrage du trading ({args.days} jours)...")
        bot.run_trading_session(days=args.days)
        
        # Rapport
        bot.print_performance_report()
        bot.save_trading_log()
        
        print("\n✅ Terminé!")
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Mode interactif si aucun argument
        interactive_mode()
    else:
        # Mode ligne de commande
        main() 