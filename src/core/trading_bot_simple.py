"""
Module de trading automatique simplifié utilisant l'analyse technique
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .technical_analysis import TechnicalAnalysis
from .data_collector import DataCollector

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTradingBot:
    """
    Bot de trading simplifié utilisant l'analyse technique
    """
    
    def __init__(self, symbol: str, initial_capital: float = 10000, 
                 risk_per_trade: float = 0.02, max_position_size: float = 0.1):
        """
        Initialise le bot de trading
        
        Args:
            symbol (str): Symbole de l'action à trader
            initial_capital (float): Capital initial
            risk_per_trade (float): Risque par trade (2% par défaut)
            max_position_size (float): Taille maximale de position (10% par défaut)
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        
        # État du trading
        self.position = 0  # Nombre d'actions détenues
        self.entry_price = 0
        self.trades = []
        self.portfolio_value = []
        self.daily_returns = []
        
        # Composants
        self.data_collector = DataCollector()
        self.technical_analyzer = None
        
        # Paramètres de trading
        self.confidence_threshold = 0.6
        self.stop_loss_pct = 0.05  # 5% de stop loss
        self.take_profit_pct = 0.15  # 15% de take profit
        self.trailing_stop_pct = 0.03  # 3% de trailing stop
        
        logger.info(f"Bot de trading simplifié initialisé pour {symbol} avec capital {initial_capital}")
    
    def prepare_data(self, period: str = "2y") -> pd.DataFrame:
        """
        Prépare les données pour l'analyse
        """
        logger.info("Préparation des données...")
        
        # Récupérer les données
        data = self.data_collector.get_stock_data(self.symbol, period=period)
        
        if data is None or len(data) < 100:
            logger.error("Données insuffisantes pour l'analyse")
            return None
        
        # Ajouter l'analyse technique
        self.technical_analyzer = TechnicalAnalysis(data)
        self.technical_analyzer.add_all_indicators()
        
        return self.technical_analyzer.data
    
    def get_simple_prediction(self, days_ahead: int = 5) -> Dict:
        """
        Obtient une prédiction simple basée sur l'analyse technique
        """
        try:
            logger.info("Obtention de la prédiction technique...")
            
            if self.technical_analyzer is None:
                return None
            
            data = self.technical_analyzer.data
            current_price = data['Close'].iloc[-1]
            
            # Calculer la tendance basée sur les moyennes mobiles
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            sma_200 = data['SMA_200'].iloc[-1]
            
            # Calculer le momentum
            momentum = (current_price / data['Close'].iloc[-5] - 1) * 100
            
            # Calculer la volatilité
            volatility = data['Close'].pct_change().rolling(20).std().iloc[-1]
            
            # Déterminer la tendance
            if current_price > sma_20 > sma_50 > sma_200:
                trend = "BULLISH"
                confidence = 0.8
            elif current_price < sma_20 < sma_50 < sma_200:
                trend = "BEARISH"
                confidence = 0.8
            elif current_price > sma_20 and sma_20 > sma_50:
                trend = "BULLISH"
                confidence = 0.6
            elif current_price < sma_20 and sma_20 < sma_50:
                trend = "BEARISH"
                confidence = 0.6
            else:
                trend = "NEUTRAL"
                confidence = 0.4
            
            # Ajuster la confiance selon le momentum
            if abs(momentum) > 5:
                confidence += 0.1
            elif abs(momentum) < 1:
                confidence -= 0.1
            
            # Générer des prédictions simples
            predictions = []
            for day in range(days_ahead):
                if trend == "BULLISH":
                    change = np.random.normal(0.001, volatility)  # Légère hausse
                elif trend == "BEARISH":
                    change = np.random.normal(-0.001, volatility)  # Légère baisse
                else:
                    change = np.random.normal(0, volatility)  # Pas de tendance
                
                if day == 0:
                    predicted_price = current_price * (1 + change)
                else:
                    predicted_price = predictions[-1] * (1 + change)
                
                predictions.append(predicted_price)
            
            return {
                'prediction': np.array(predictions),
                'confidence': min(confidence, 0.95),
                'trend': trend,
                'momentum': momentum,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return None
    
    def get_technical_signals(self) -> Dict:
        """
        Obtient les signaux techniques
        """
        if self.technical_analyzer is None:
            return {}
        
        signals = self.technical_analyzer.get_signals()
        
        # Calculer un score technique global
        technical_score = 0
        signal_count = 0
        
        for signal_type, signal_value in signals.items():
            if signal_value == 'BUY':
                technical_score += 1
            elif signal_value == 'SELL':
                technical_score -= 1
            signal_count += 1
        
        if signal_count > 0:
            technical_score = technical_score / signal_count
        
        return {
            'signals': signals,
            'technical_score': technical_score,
            'overall_signal': 'BUY' if technical_score > 0.2 else 'SELL' if technical_score < -0.2 else 'HOLD'
        }
    
    def calculate_position_size(self, entry_price: float, confidence: float) -> int:
        """
        Calcule la taille de position optimale
        """
        # Capital disponible pour ce trade
        available_capital = self.current_capital * self.max_position_size
        
        # Ajuster selon la confiance
        adjusted_capital = available_capital * confidence
        
        # Calculer le nombre d'actions
        position_size = int(adjusted_capital / entry_price)
        
        # Limiter selon le risque
        max_risk_capital = self.current_capital * self.risk_per_trade
        max_shares_by_risk = int(max_risk_capital / (entry_price * self.stop_loss_pct))
        
        return min(position_size, max_shares_by_risk)
    
    def should_buy(self, current_price: float, prediction: Dict, technical_signals: Dict) -> Tuple[bool, float]:
        """
        Détermine s'il faut acheter
        """
        # Conditions d'achat
        conditions = []
        
        # 1. Prédiction bullish avec confiance élevée
        if prediction and prediction['trend'] == 'BULLISH' and prediction['confidence'] > self.confidence_threshold:
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 2. Signaux techniques favorables
        if technical_signals.get('overall_signal') == 'BUY':
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 3. Pas de position actuelle
        if self.position == 0:
            conditions.append(True)
        else:
            conditions.append(False)
        
        # 4. Capital suffisant
        if self.current_capital > current_price * 100:  # Au moins 100 actions
            conditions.append(True)
        else:
            conditions.append(False)
        
        # Décision finale
        should_buy = sum(conditions) >= 3  # Au moins 3 conditions sur 4
        
        # Calculer la confiance de la décision
        decision_confidence = sum(conditions) / len(conditions)
        
        return should_buy, decision_confidence
    
    def should_sell(self, current_price: float, prediction: Dict, technical_signals: Dict) -> Tuple[bool, float]:
        """
        Détermine s'il faut vendre
        """
        if self.position == 0:
            return False, 0.0
        
        # Conditions de vente
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
        
        # Décision finale
        should_sell = sum(conditions) >= 2  # Au moins 2 conditions sur 5
        
        # Calculer la confiance de la décision
        decision_confidence = sum(conditions) / len(conditions)
        
        return should_sell, decision_confidence
    
    def execute_buy(self, current_price: float, confidence: float):
        """
        Exécute un ordre d'achat
        """
        position_size = self.calculate_position_size(current_price, confidence)
        
        if position_size > 0:
            cost = position_size * current_price
            commission = cost * 0.001  # 0.1% de commission
            
            if cost + commission <= self.current_capital:
                self.position = position_size
                self.entry_price = current_price
                self.current_capital -= (cost + commission)
                self.highest_price = current_price
                
                trade = {
                    'type': 'BUY',
                    'date': datetime.now(),
                    'price': current_price,
                    'shares': position_size,
                    'cost': cost,
                    'commission': commission,
                    'confidence': confidence
                }
                self.trades.append(trade)
                
                logger.info(f"ACHAT: {position_size} actions à {current_price:.2f} (confiance: {confidence:.2f})")
                return True
        
        return False
    
    def execute_sell(self, current_price: float, confidence: float):
        """
        Exécute un ordre de vente
        """
        if self.position > 0:
            revenue = self.position * current_price
            commission = revenue * 0.001  # 0.1% de commission
            
            self.current_capital += (revenue - commission)
            
            # Calculer le P&L
            pnl = revenue - (self.position * self.entry_price) - commission
            
            trade = {
                'type': 'SELL',
                'date': datetime.now(),
                'price': current_price,
                'shares': self.position,
                'revenue': revenue,
                'commission': commission,
                'pnl': pnl,
                'confidence': confidence
            }
            self.trades.append(trade)
            
            logger.info(f"VENTE: {self.position} actions à {current_price:.2f} (P&L: {pnl:.2f}, confiance: {confidence:.2f})")
            
            # Réinitialiser la position
            self.position = 0
            self.entry_price = 0
            self.highest_price = 0
            
            return True
        
        return False
    
    def update_portfolio_value(self, current_price: float):
        """
        Met à jour la valeur du portefeuille
        """
        portfolio_value = self.current_capital + (self.position * current_price)
        self.portfolio_value.append({
            'date': datetime.now(),
            'value': portfolio_value,
            'cash': self.current_capital,
            'position_value': self.position * current_price,
            'shares': self.position
        })
        
        # Calculer le rendement quotidien
        if len(self.portfolio_value) > 1:
            daily_return = (portfolio_value - self.portfolio_value[-2]['value']) / self.portfolio_value[-2]['value']
            self.daily_returns.append(daily_return)
    
    def run_trading_session(self, days: int = 30):
        """
        Exécute une session de trading
        """
        logger.info(f"Début de la session de trading pour {days} jours...")
        
        # Préparer les données
        data = self.prepare_data()
        if data is None:
            logger.error("Impossible de préparer les données")
            return
        
        # Simuler le trading jour par jour
        for i in range(min(days, len(data))):
            current_data = data.iloc[-days+i:]
            current_price = current_data['Close'].iloc[-1]
            
            # Mettre à jour le prix le plus haut pour le trailing stop
            if hasattr(self, 'highest_price') and self.position > 0:
                self.highest_price = max(self.highest_price, current_price)
            
            # Obtenir les prédictions et signaux
            prediction = self.get_simple_prediction()
            technical_signals = self.get_technical_signals()
            
            # Décisions de trading
            if self.position == 0:  # Pas de position
                should_buy, buy_confidence = self.should_buy(current_price, prediction, technical_signals)
                if should_buy:
                    self.execute_buy(current_price, buy_confidence)
            else:  # Position ouverte
                should_sell, sell_confidence = self.should_sell(current_price, prediction, technical_signals)
                if should_sell:
                    self.execute_sell(current_price, sell_confidence)
            
            # Mettre à jour la valeur du portefeuille
            self.update_portfolio_value(current_price)
        
        logger.info("Session de trading terminée")
    
    def get_performance_metrics(self) -> Dict:
        """
        Calcule les métriques de performance
        """
        if not self.portfolio_value:
            return {}
        
        initial_value = self.initial_capital
        final_value = self.portfolio_value[-1]['value']
        total_return = (final_value - initial_value) / initial_value
        
        # Calculer les métriques
        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_value': final_value,
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t.get('pnl', 0) > 0]),
            'losing_trades': len([t for t in self.trades if t.get('pnl', 0) < 0]),
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        # Win rate
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
        
        # Moyennes des gains/pertes
        wins = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t['pnl'] for t in self.trades if t.get('pnl', 0) < 0]
        
        if wins:
            metrics['avg_win'] = np.mean(wins)
        if losses:
            metrics['avg_loss'] = np.mean(losses)
        
        # Maximum drawdown
        if self.portfolio_value:
            values = [pv['value'] for pv in self.portfolio_value]
            peak = values[0]
            max_dd = 0
            
            for value in values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            
            metrics['max_drawdown'] = max_dd
        
        # Ratio de Sharpe
        if self.daily_returns:
            returns_array = np.array(self.daily_returns)
            if len(returns_array) > 1:
                metrics['sharpe_ratio'] = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        
        return metrics
    
    def print_performance_report(self):
        """
        Affiche un rapport de performance
        """
        metrics = self.get_performance_metrics()
        
        if not metrics:
            logger.info("Aucune donnée de performance disponible")
            return
        
        print("\n" + "="*50)
        print("RAPPORT DE PERFORMANCE DU BOT DE TRADING")
        print("="*50)
        print(f"Symbole: {self.symbol}")
        print(f"Capital initial: {self.initial_capital:,.2f} €")
        print(f"Valeur finale: {metrics['final_value']:,.2f} €")
        print(f"Rendement total: {metrics['total_return_pct']:.2f}%")
        print(f"Nombre de trades: {metrics['total_trades']}")
        print(f"Trades gagnants: {metrics['winning_trades']}")
        print(f"Trades perdants: {metrics['losing_trades']}")
        print(f"Taux de réussite: {metrics['win_rate']:.2%}")
        print(f"Gain moyen: {metrics['avg_win']:,.2f} €")
        print(f"Perte moyenne: {metrics['avg_loss']:,.2f} €")
        print(f"Drawdown maximum: {metrics['max_drawdown']:.2%}")
        print(f"Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
        print("="*50)
    
    def save_trading_log(self, filename: str = None):
        """
        Sauvegarde le journal de trading
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_log_{self.symbol}_{timestamp}.csv"
        
        # Créer un DataFrame avec les trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(filename, index=False)
            logger.info(f"Journal de trading sauvegardé: {filename}")
        
        # Sauvegarder l'évolution du portefeuille
        if self.portfolio_value:
            portfolio_df = pd.DataFrame(self.portfolio_value)
            portfolio_filename = filename.replace('.csv', '_portfolio.csv')
            portfolio_df.to_csv(portfolio_filename, index=False)
            logger.info(f"Évolution du portefeuille sauvegardée: {portfolio_filename}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer et exécuter le bot de trading
    bot = SimpleTradingBot('AAPL', initial_capital=10000)
    
    # Exécuter une session de trading de 30 jours
    bot.run_trading_session(days=30)
    
    # Afficher le rapport de performance
    bot.print_performance_report()
    
    # Sauvegarder les résultats
    bot.save_trading_log() 