"""
Système de trading automatique complet avec interface utilisateur
"""

import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional

from core.trading_bot_simple import SimpleTradingBot
from core.data_collector import DataCollector
from core.technical_analysis import TechnicalAnalysis

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'../logs/trading_system_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystem:
    """
    Système de trading complet avec interface utilisateur
    """
    
    def __init__(self):
        self.bots = {}
        self.portfolio = {}
        
    def create_bot(self, symbol: str, capital: float, risk: float = 0.02, 
                   position_size: float = 0.1, aggressive: bool = False) -> SimpleTradingBot:
        """
        Crée un nouveau bot de trading
        """
        bot = SimpleTradingBot(
            symbol=symbol,
            initial_capital=capital,
            risk_per_trade=risk,
            max_position_size=position_size
        )
        
        if aggressive:
            # Paramètres agressifs
            bot.confidence_threshold = 0.1
            bot.stop_loss_pct = 0.02
            bot.take_profit_pct = 0.05
            bot.trailing_stop_pct = 0.01
        
        self.bots[symbol] = bot
        logger.info(f"Bot créé pour {symbol} avec capital {capital:,.2f} €")
        
        return bot
    
    def run_analysis(self, symbol: str) -> Dict:
        """
        Lance une analyse complète pour un symbole
        """
        print(f"\n📊 ANALYSE COMPLÈTE POUR {symbol}")
        print("=" * 50)
        
        # Récupérer les données
        collector = DataCollector()
        data = collector.get_stock_data(symbol, period="2y")
        
        if data is None:
            print(f"❌ Impossible de récupérer les données pour {symbol}")
            return {}
        
        print(f"✅ Données récupérées: {len(data)} enregistrements")
        
        # Analyse technique
        analyzer = TechnicalAnalysis(data)
        analyzer.add_all_indicators()
        
        # Informations actuelles
        current_price = data['Close'].iloc[-1]
        print(f"💰 Prix actuel: {current_price:.2f} €")
        
        # Indicateurs techniques (utiliser les données de l'analyseur)
        analyzed_data = analyzer.data
        print(f"\n📈 Indicateurs techniques:")
        
        # Vérifier si les colonnes existent avant de les afficher
        if 'RSI' in analyzed_data.columns:
            print(f"  RSI: {analyzed_data['RSI'].iloc[-1]:.2f}")
        if 'MACD' in analyzed_data.columns:
            print(f"  MACD: {analyzed_data['MACD'].iloc[-1]:.4f}")
        if 'SMA_20' in analyzed_data.columns:
            print(f"  SMA 20: {analyzed_data['SMA_20'].iloc[-1]:.2f}")
        if 'SMA_50' in analyzed_data.columns:
            print(f"  SMA 50: {analyzed_data['SMA_50'].iloc[-1]:.2f}")
        if 'SMA_200' in analyzed_data.columns:
            print(f"  SMA 200: {analyzed_data['SMA_200'].iloc[-1]:.2f}")
        
        # Signaux techniques
        signals = analyzer.get_signals()
        print(f"\n🎯 Signaux techniques:")
        for signal_type, signal_value in signals.items():
            print(f"  {signal_type}: {signal_value}")
        
        # Informations sur l'entreprise
        company_info = collector.get_company_info(symbol)
        if company_info:
            print(f"\n🏢 Informations sur l'entreprise:")
            for key, value in company_info.items():
                print(f"  {key}: {value}")
        
        return {
            'data': analyzed_data,
            'signals': signals,
            'company_info': company_info,
            'current_price': current_price
        }
    
    def run_trading_session(self, symbol: str, days: int = 30, 
                          capital: float = 10000, aggressive: bool = False) -> Dict:
        """
        Lance une session de trading
        """
        print(f"\n🔄 SESSION DE TRADING POUR {symbol}")
        print("=" * 50)
        
        # Créer ou récupérer le bot
        if symbol not in self.bots:
            self.create_bot(symbol, capital, aggressive=aggressive)
        
        bot = self.bots[symbol]
        
        # Configuration
        print(f"📊 Configuration:")
        print(f"  Capital: {bot.initial_capital:,.2f} €")
        print(f"  Risque par trade: {bot.risk_per_trade*100:.1f}%")
        print(f"  Position max: {bot.max_position_size*100:.1f}%")
        print(f"  Seuil confiance: {bot.confidence_threshold}")
        print(f"  Stop loss: {bot.stop_loss_pct*100:.1f}%")
        print(f"  Take profit: {bot.take_profit_pct*100:.1f}%")
        
        # Lancer la session
        bot.run_trading_session(days=days)
        
        # Rapport de performance
        metrics = bot.get_performance_metrics()
        
        print(f"\n📊 RÉSULTATS DE LA SESSION")
        print("=" * 50)
        print(f"Rendement: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Trades: {metrics.get('total_trades', 0)}")
        print(f"Taux de réussite: {metrics.get('win_rate', 0):.2%}")
        print(f"Gain moyen: {metrics.get('avg_win', 0):,.2f} €")
        print(f"Perte moyenne: {metrics.get('avg_loss', 0):,.2f} €")
        
        # Sauvegarder les résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_session_{symbol}_{timestamp}"
        bot.save_trading_log(filename)
        
        return metrics
    
    def run_portfolio_simulation(self, symbols: List[str], capital_per_symbol: float = 5000,
                               days: int = 30, aggressive: bool = False) -> Dict:
        """
        Lance une simulation de portefeuille multi-actifs
        """
        print(f"\n🌐 SIMULATION DE PORTEFEUILLE MULTI-ACTIFS")
        print("=" * 60)
        
        total_capital = len(symbols) * capital_per_symbol
        print(f"Capital total: {total_capital:,.2f} €")
        print(f"Capital par action: {capital_per_symbol:,.2f} €")
        print(f"Actions: {', '.join(symbols)}")
        
        portfolio_results = {}
        total_return = 0
        total_trades = 0
        
        for symbol in symbols:
            print(f"\n📈 Traitement de {symbol}...")
            
            try:
                metrics = self.run_trading_session(
                    symbol=symbol,
                    days=days,
                    capital=capital_per_symbol,
                    aggressive=aggressive
                )
                
                portfolio_results[symbol] = metrics
                total_return += metrics.get('total_return', 0) * capital_per_symbol
                total_trades += metrics.get('total_trades', 0)
                
            except Exception as e:
                print(f"❌ Erreur pour {symbol}: {str(e)}")
                continue
        
        # Résumé du portefeuille
        print(f"\n📊 RÉSUMÉ DU PORTEFEUILLE")
        print("=" * 60)
        print(f"Capital total: {total_capital:,.2f} €")
        print(f"Rendement total: {(total_return/total_capital)*100:.2f}%")
        print(f"Nombre total de trades: {total_trades}")
        
        for symbol, metrics in portfolio_results.items():
            print(f"{symbol}: {metrics.get('total_return_pct', 0):.2f}% ({metrics.get('total_trades', 0)} trades)")
        
        return portfolio_results
    
    def interactive_mode(self):
        """
        Mode interactif pour l'utilisateur
        """
        print("🎮 MODE INTERACTIF - SYSTÈME DE TRADING")
        print("=" * 50)
        
        while True:
            print(f"\nOptions disponibles:")
            print("1. Analyser une action")
            print("2. Lancer une session de trading")
            print("3. Simulation de portefeuille")
            print("4. Voir les bots actifs")
            print("5. Quitter")
            
            choice = input("\nVotre choix (1-5): ").strip()
            
            if choice == '1':
                symbol = input("Symbole de l'action (ex: AAPL): ").strip().upper()
                if symbol:
                    self.run_analysis(symbol)
            
            elif choice == '2':
                symbol = input("Symbole de l'action: ").strip().upper()
                if symbol:
                    try:
                        capital = float(input("Capital (défaut: 10000): ") or "10000")
                        days = int(input("Nombre de jours (défaut: 30): ") or "30")
                        aggressive = input("Mode agressif? (oui/non): ").strip().lower() in ['oui', 'yes', 'o', 'y']
                        
                        self.run_trading_session(symbol, days, capital, aggressive)
                    except ValueError:
                        print("❌ Valeur invalide")
            
            elif choice == '3':
                symbols_input = input("Symboles séparés par des virgules (ex: AAPL,MSFT,TSLA): ").strip()
                if symbols_input:
                    symbols = [s.strip().upper() for s in symbols_input.split(',')]
                    try:
                        capital_per_symbol = float(input("Capital par action (défaut: 5000): ") or "5000")
                        days = int(input("Nombre de jours (défaut: 30): ") or "30")
                        aggressive = input("Mode agressif? (oui/non): ").strip().lower() in ['oui', 'yes', 'o', 'y']
                        
                        self.run_portfolio_simulation(symbols, capital_per_symbol, days, aggressive)
                    except ValueError:
                        print("❌ Valeur invalide")
            
            elif choice == '4':
                if self.bots:
                    print(f"\n🤖 Bots actifs:")
                    for symbol, bot in self.bots.items():
                        print(f"  {symbol}: {bot.initial_capital:,.2f} €")
                else:
                    print("Aucun bot actif")
            
            elif choice == '5':
                print("👋 Au revoir!")
                break
            
            else:
                print("❌ Choix invalide")

def main():
    """
    Fonction principale avec interface en ligne de commande
    """
    parser = argparse.ArgumentParser(description='Système de trading automatique complet')
    parser.add_argument('--symbol', '-s', type=str, help='Symbole de l\'action à analyser')
    parser.add_argument('--capital', '-c', type=float, default=10000, help='Capital initial')
    parser.add_argument('--days', '-d', type=int, default=30, help='Nombre de jours de trading')
    parser.add_argument('--aggressive', '-a', action='store_true', help='Mode agressif')
    parser.add_argument('--portfolio', '-p', type=str, help='Symboles pour portefeuille (séparés par des virgules)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Mode interactif')
    
    args = parser.parse_args()
    
    # Créer le système
    system = TradingSystem()
    
    if args.interactive or len(sys.argv) == 1:
        # Mode interactif
        system.interactive_mode()
    elif args.portfolio:
        # Mode portefeuille
        symbols = [s.strip().upper() for s in args.portfolio.split(',')]
        system.run_portfolio_simulation(symbols, args.capital, args.days, args.aggressive)
    elif args.symbol:
        # Mode action unique
        if args.symbol.upper() == 'ANALYSE':
            # Analyser plusieurs actions populaires
            popular_symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN']
            for symbol in popular_symbols:
                system.run_analysis(symbol)
        else:
            # Analyser une action spécifique
            system.run_analysis(args.symbol)
            system.run_trading_session(args.symbol, args.days, args.capital, args.aggressive)
    else:
        # Aide
        parser.print_help()

if __name__ == "__main__":
    main() 