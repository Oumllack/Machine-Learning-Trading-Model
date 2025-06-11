"""
Démonstration agressive du bot de trading avec paramètres très permissifs
"""

import sys
import logging
from typing import Dict, Tuple
from trading_bot_simple import SimpleTradingBot

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AggressiveTradingBot(SimpleTradingBot):
    """
    Version agressive du bot de trading avec paramètres très permissifs
    """
    
    def __init__(self, symbol: str, initial_capital: float = 10000):
        super().__init__(symbol, initial_capital, risk_per_trade=0.1, max_position_size=0.3)
        
        # Paramètres très permissifs
        self.confidence_threshold = 0.1  # Très bas
        self.stop_loss_pct = 0.02  # Stop loss très serré
        self.take_profit_pct = 0.05  # Take profit rapide
        self.trailing_stop_pct = 0.01  # Trailing stop très serré
        
        logger.info(f"Bot de trading agressif initialisé pour {symbol}")
    
    def should_buy(self, current_price: float, prediction: Dict, technical_signals: Dict) -> Tuple[bool, float]:
        """
        Conditions d'achat très permissives
        """
        # Conditions d'achat simplifiées
        conditions = []
        
        # 1. Prédiction bullish OU neutre
        if prediction and prediction['trend'] in ['BULLISH', 'NEUTRAL']:
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 2. Signaux techniques favorables OU neutres
        if technical_signals.get('overall_signal') in ['BUY', 'HOLD']:
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 3. Pas de position actuelle
        if self.position == 0:
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 4. Capital suffisant
        if self.current_capital > current_price * 10:  # Au moins 10 actions
            conditions.append(True)
        else:
            conditions.append(False)
        
        # Décision finale - très permissive
        should_buy = sum(conditions) >= 2  # Seulement 2 conditions sur 4
        
        # Confiance basée sur le nombre de conditions
        decision_confidence = sum(conditions) / len(conditions)
        
        return should_buy, decision_confidence
    
    def should_sell(self, current_price: float, prediction: Dict, technical_signals: Dict) -> Tuple[bool, float]:
        """
        Conditions de vente très permissives
        """
        if self.position == 0:
            return False, 0.0
        
        # Conditions de vente simplifiées
        conditions = []
        
        # 1. Prédiction bearish
        if prediction and prediction['trend'] == 'BEARISH':
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 2. Signaux techniques de vente
        if technical_signals.get('overall_signal') == 'SELL':
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 3. Stop loss
        if current_price <= self.entry_price * (1 - self.stop_loss_pct):
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 4. Take profit
        if current_price >= self.entry_price * (1 + self.take_profit_pct):
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 5. Trailing stop
        if hasattr(self, 'highest_price') and current_price <= self.highest_price * (1 - self.trailing_stop_pct):
            conditions.append(True)
        else:
            conditions.append(False)
        
        # Décision finale - très permissive
        should_sell = sum(conditions) >= 1  # Seulement 1 condition sur 5
        
        # Confiance basée sur le nombre de conditions
        decision_confidence = sum(conditions) / len(conditions)
        
        return should_sell, decision_confidence

def demo_aggressive_trading():
    """
    Démonstration avec le bot agressif
    """
    print("🚀 DÉMONSTRATION AGRESSIVE DU BOT DE TRADING")
    print("=" * 60)
    
    # Créer le bot agressif
    bot = AggressiveTradingBot('AAPL', initial_capital=10000)
    
    print(f"📊 Configuration du bot agressif:")
    print(f"  Symbole: {bot.symbol}")
    print(f"  Capital: {bot.initial_capital:,.2f} €")
    print(f"  Risque par trade: {bot.risk_per_trade*100:.1f}%")
    print(f"  Position max: {bot.max_position_size*100:.1f}%")
    print(f"  Seuil confiance: {bot.confidence_threshold}")
    print(f"  Stop loss: {bot.stop_loss_pct*100:.1f}%")
    print(f"  Take profit: {bot.take_profit_pct*100:.1f}%")
    print(f"  Trailing stop: {bot.trailing_stop_pct*100:.1f}%")
    
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
    
    # Obtenir les signaux
    technical_signals = bot.get_technical_signals()
    print(f"\n🎯 Signaux techniques:")
    print(f"  Signal global: {technical_signals.get('overall_signal', 'N/A')}")
    print(f"  Score technique: {technical_signals.get('technical_score', 0):.3f}")
    
    # Obtenir la prédiction
    prediction = bot.get_simple_prediction(days_ahead=5)
    if prediction:
        print(f"\n🧠 Prédiction technique:")
        print(f"  Tendance: {prediction['trend']}")
        print(f"  Confiance: {prediction['confidence']:.2f}")
        print(f"  Momentum: {prediction['momentum']:.2f}%")
    
    # Simuler le trading sur une période plus longue
    print(f"\n🔄 Simulation de trading agressive (90 jours)...")
    bot.run_trading_session(days=90)
    
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
    else:
        print(f"\n⚠️  Aucun trade effectué - conditions encore trop strictes")
    
    # Sauvegarder les résultats
    bot.save_trading_log("aggressive_trading_results")
    print(f"💾 Résultats sauvegardés dans aggressive_trading_results.csv")
    
    print(f"\n✅ Démonstration agressive terminée!")

def demo_manual_trading():
    """
    Démonstration avec trading manuel simulé
    """
    print("🎮 DÉMONSTRATION DE TRADING MANUEL SIMULÉ")
    print("=" * 60)
    
    # Créer le bot
    bot = AggressiveTradingBot('AAPL', initial_capital=10000)
    
    # Préparer les données
    data = bot.prepare_data(period="1y")
    if data is None:
        print("❌ Impossible de récupérer les données")
        return
    
    print(f"✅ Données récupérées: {len(data)} enregistrements")
    
    # Simuler des trades manuels
    print(f"\n🔄 Simulation de trades manuels...")
    
    # Trade 1: Achat
    current_price = data['Close'].iloc[-1]
    print(f"📈 Achat simulé à {current_price:.2f} €")
    bot.execute_buy(current_price, 0.8)
    
    # Simuler une hausse
    new_price = current_price * 1.03  # +3%
    print(f"📈 Prix monte à {new_price:.2f} € (+3%)")
    bot.update_portfolio_value(new_price)
    
    # Trade 2: Vente avec profit
    print(f"📉 Vente simulée à {new_price:.2f} €")
    bot.execute_sell(new_price, 0.9)
    
    # Trade 3: Achat à nouveau
    new_price2 = new_price * 0.98  # -2%
    print(f"📈 Nouvel achat à {new_price2:.2f} €")
    bot.execute_buy(new_price2, 0.7)
    
    # Simuler une baisse
    new_price3 = new_price2 * 0.97  # -3%
    print(f"📉 Prix baisse à {new_price3:.2f} € (-3%)")
    bot.update_portfolio_value(new_price3)
    
    # Trade 4: Vente avec perte (stop loss)
    print(f"📉 Vente avec stop loss à {new_price3:.2f} €")
    bot.execute_sell(new_price3, 0.6)
    
    # Afficher le rapport
    print(f"\n" + "="*60)
    bot.print_performance_report()
    
    # Détails des trades
    if bot.trades:
        print(f"\n📋 Détail des trades manuels:")
        for i, trade in enumerate(bot.trades, 1):
            print(f"  Trade {i}:")
            print(f"    Type: {trade['type']}")
            print(f"    Prix: {trade['price']:.2f} €")
            print(f"    Actions: {trade['shares']}")
            if 'pnl' in trade:
                print(f"    P&L: {trade['pnl']:.2f} €")
            print()
    
    # Sauvegarder
    bot.save_trading_log("manual_trading_results")
    print(f"💾 Résultats sauvegardés dans manual_trading_results.csv")

def main():
    """
    Fonction principale
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == 'manual':
            demo_manual_trading()
        else:
            demo_aggressive_trading()
    else:
        demo_aggressive_trading()

if __name__ == "__main__":
    main() 