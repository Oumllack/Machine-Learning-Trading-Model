"""
Script principal pour l'analyse boursière et prédiction de prix
Combine récupération de données, analyse technique et LSTM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging

# Import des modules personnalisés
try:
    from lstm_predictor import LSTMPredictor
    LSTM_AVAILABLE = True
except ImportError:
    from lstm_pytorch import LSTMPyTorchPredictor
    LSTM_AVAILABLE = False
from data_collector import DataCollector
from technical_analysis import TechnicalAnalysis

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockAnalyzer:
    """
    Classe principale pour l'analyse boursière complète
    """
    
    def __init__(self):
        self.collector = DataCollector()
        self.ta_analyzer = None
        self.lstm_predictor = None
        self.data = None
        
    def analyze_stock(self, 
                     symbol: str,
                     period: str = "2y",
                     sequence_length: int = 60,
                     lstm_units: list = [50, 50],
                     epochs: int = 50) -> dict:
        """
        Analyse complète d'une action
        
        Args:
            symbol (str): Symbole de l'action
            period (str): Période des données
            sequence_length (int): Longueur de séquence LSTM
            lstm_units (list): Unités LSTM
            epochs (int): Nombre d'époques d'entraînement
            
        Returns:
            dict: Résultats de l'analyse
        """
        logger.info(f"Début de l'analyse pour {symbol}")
        
        # 1. Récupération des données
        logger.info("Étape 1: Récupération des données...")
        self.data = self.collector.get_stock_data(symbol, period=period)
        
        if self.data is None:
            logger.error(f"Impossible de récupérer les données pour {symbol}")
            return None
        
        # 2. Analyse technique
        logger.info("Étape 2: Analyse technique...")
        self.ta_analyzer = TechnicalAnalysis(self.data)
        self.ta_analyzer.add_all_indicators()
        
        # Obtenir les signaux techniques
        technical_signals = self.ta_analyzer.get_signals()
        technical_stats = self.ta_analyzer.get_summary_stats()
        
        # 3. Prédiction LSTM
        logger.info("Étape 3: Prédiction LSTM...")
        if LSTM_AVAILABLE:
            self.lstm_predictor = LSTMPredictor(self.ta_analyzer.data)
            X_train, y_train, X_val, y_val, X_test, y_test = self.lstm_predictor.prepare_data(
                sequence_length=sequence_length,
                test_size=0.2,
                validation_size=0.1
            )
            self.lstm_predictor.build_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                lstm_units=lstm_units,
                dropout_rate=0.2
            )
            history = self.lstm_predictor.train_model(
                X_train, y_train, X_val, y_val,
                epochs=epochs,
                batch_size=32
            )
        else:
            self.lstm_predictor = LSTMPyTorchPredictor(self.ta_analyzer.data)
            X_train, y_train, X_val, y_val, X_test, y_test = self.lstm_predictor.prepare_data(
                sequence_length=sequence_length,
                test_size=0.2,
                validation_size=0.1
            )
            self.lstm_predictor.build_model(input_size=X_train.shape[2])
            self.lstm_predictor.train_model(
                X_train, y_train, X_val, y_val,
                epochs=epochs
            )
        
        # Évaluer le modèle
        lstm_metrics = self.lstm_predictor.evaluate_model(X_test, y_test)
        
        # Prédictions futures
        future_predictions = self.lstm_predictor.predict_future(days_ahead=30)
        
        # 4. Compiler les résultats
        results = {
            'symbol': symbol,
            'data_points': len(self.data),
            'current_price': self.data['Close'].iloc[-1],
            'technical_signals': technical_signals,
            'technical_stats': technical_stats,
            'lstm_metrics': lstm_metrics,
            'future_predictions': future_predictions,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Analyse terminée pour {symbol}")
        return results
    
    def analyze_oil(self, oil_type: str = "BZ=F") -> dict:
        """
        Analyse du pétrole (Brent ou WTI)
        
        Args:
            oil_type (str): Type de pétrole ('BZ=F' pour Brent, 'CL=F' pour WTI)
            
        Returns:
            dict: Résultats de l'analyse
        """
        oil_names = {
            'BZ=F': 'Brent Crude Oil',
            'CL=F': 'WTI Crude Oil'
        }
        
        symbol_name = oil_names.get(oil_type, oil_type)
        logger.info(f"Analyse du pétrole: {symbol_name}")
        
        return self.analyze_stock(oil_type, period="2y")
    
    def compare_stocks(self, symbols: list, period: str = "1y") -> dict:
        """
        Compare plusieurs actions
        
        Args:
            symbols (list): Liste des symboles à comparer
            period (str): Période des données
            
        Returns:
            dict: Résultats de comparaison
        """
        logger.info(f"Comparaison de {len(symbols)} actions")
        
        comparison_results = {}
        
        for symbol in symbols:
            logger.info(f"Analyse de {symbol}...")
            result = self.analyze_stock(symbol, period=period, epochs=30)  # Moins d'époques pour la comparaison
            if result:
                comparison_results[symbol] = result
        
        return comparison_results
    
    def generate_report(self, results: dict, save_plots: bool = True) -> str:
        """
        Génère un rapport d'analyse
        
        Args:
            results (dict): Résultats de l'analyse
            save_plots (bool): Sauvegarder les graphiques
            
        Returns:
            str: Rapport formaté
        """
        if results is None:
            return "Aucun résultat à analyser"
        
        symbol = results['symbol']
        
        # Créer le rapport
        report = f"""
=== RAPPORT D'ANALYSE BOURSIÈRE ===
Symbole: {symbol}
Date d'analyse: {results['analysis_date']}
Points de données: {results['data_points']:,}
Prix actuel: ${results['current_price']:.2f}

--- SIGNAUX TECHNIQUES ---
"""
        
        for indicator, signal in results['technical_signals'].items():
            report += f"{indicator}: {signal}\n"
        
        report += f"""
--- STATISTIQUES TECHNIQUES ---
"""
        
        for indicator, stats in results['technical_stats'].items():
            report += f"{indicator}:\n"
            for stat, value in stats.items():
                report += f"  {stat}: {value}\n"
        
        report += f"""
--- PERFORMANCES LSTM ---
"""
        
        for metric, value in results['lstm_metrics'].items():
            report += f"{metric}: {value:.4f}\n"
        
        report += f"""
--- PRÉDICTIONS FUTURES (30 jours) ---
Prix prédit dans 7 jours: ${results['future_predictions'][6]:.2f}
Prix prédit dans 14 jours: ${results['future_predictions'][13]:.2f}
Prix prédit dans 30 jours: ${results['future_predictions'][-1]:.2f}
Tendance prédite: {'Haussière' if results['future_predictions'][-1] > results['current_price'] else 'Baissière'}
"""
        
        # Afficher les graphiques
        if self.ta_analyzer:
            self.ta_analyzer.plot_indicators(symbol)
        
        if self.lstm_predictor:
            self.lstm_predictor.plot_training_history()
            self.lstm_predictor.plot_predictions(symbol)
        
        # Sauvegarder le rapport
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"rapport_{symbol}_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Rapport sauvegardé dans {report_filename}")
        
        return report
    
    def save_data(self, format: str = 'csv'):
        """
        Sauvegarde les données analysées
        
        Args:
            format (str): Format de sauvegarde
        """
        if self.data is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"donnees_analysees_{timestamp}"
            
            if format == 'csv':
                self.data.to_csv(f"{filename}.csv")
            elif format == 'parquet':
                self.data.to_parquet(f"{filename}.parquet")
            
            logger.info(f"Données sauvegardées dans {filename}.{format}")

def main():
    """
    Fonction principale
    """
    print("=== ANALYSE BOURSIÈRE ET PRÉDICTION DE PRIX ===")
    print("1. Analyser une action")
    print("2. Analyser le pétrole")
    print("3. Comparer plusieurs actions")
    print("4. Exemple avec Apple (AAPL)")
    
    choice = input("\nChoisissez une option (1-4): ").strip()
    
    analyzer = StockAnalyzer()
    
    if choice == "1":
        symbol = input("Entrez le symbole de l'action (ex: AAPL, MSFT, TSLA): ").strip().upper()
        results = analyzer.analyze_stock(symbol)
        if results:
            report = analyzer.generate_report(results)
            print(report)
    
    elif choice == "2":
        oil_type = input("Type de pétrole (BZ=F pour Brent, CL=F pour WTI): ").strip().upper()
        if not oil_type:
            oil_type = "BZ=F"
        results = analyzer.analyze_oil(oil_type)
        if results:
            report = analyzer.generate_report(results)
            print(report)
    
    elif choice == "3":
        symbols_input = input("Entrez les symboles séparés par des virgules (ex: AAPL,MSFT,TSLA): ").strip()
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        results = analyzer.compare_stocks(symbols)
        
        print("\n=== RÉSULTATS DE COMPARAISON ===")
        for symbol, result in results.items():
            print(f"\n{symbol}:")
            print(f"  Prix actuel: ${result['current_price']:.2f}")
            print(f"  RSI: {result['technical_stats']['RSI']['Actuel']}")
            print(f"  RMSE LSTM: {result['lstm_metrics']['RMSE']:.4f}")
    
    elif choice == "4":
        print("Exemple avec Apple (AAPL)...")
        results = analyzer.analyze_stock('AAPL', epochs=30)  # Moins d'époques pour l'exemple
        if results:
            report = analyzer.generate_report(results)
            print(report)
    
    else:
        print("Option invalide")

if __name__ == "__main__":
    main() 