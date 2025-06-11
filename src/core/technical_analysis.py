"""
Module d'analyse technique pour les données boursières
Inclut MACD, RSI, moyennes mobiles, et autres indicateurs
"""

import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """
    Classe pour l'analyse technique des données boursières
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialise l'analyseur technique
        
        Args:
            data (pd.DataFrame): DataFrame avec les données OHLCV
        """
        self.data = data.copy()
        self.indicators = {}
        
    def add_all_indicators(self):
        """
        Ajoute tous les indicateurs techniques principaux
        """
        logger.info("Ajout de tous les indicateurs techniques...")
        
        # Moyennes mobiles
        self.add_moving_averages()
        
        # Indicateurs de momentum
        self.add_rsi()
        self.add_macd()
        self.add_stochastic()
        self.add_williams_r()
        
        # Indicateurs de volatilité
        self.add_bollinger_bands()
        self.add_atr()
        
        # Indicateurs de volume
        self.add_volume_indicators()
        
        # Indicateurs de tendance
        self.add_adx()
        self.add_ichimoku()
        
        logger.info("Tous les indicateurs ont été ajoutés")
        
    def add_moving_averages(self, periods: List[int] = [5, 10, 20, 50, 200]):
        """
        Ajoute les moyennes mobiles simples et exponentielles
        
        Args:
            periods (list): Liste des périodes pour les moyennes mobiles
        """
        for period in periods:
            # Moyenne mobile simple
            self.data[f'SMA_{period}'] = ta.trend.sma_indicator(
                self.data['Close'], window=period
            )
            
            # Moyenne mobile exponentielle
            self.data[f'EMA_{period}'] = ta.trend.ema_indicator(
                self.data['Close'], window=period
            )
        
        logger.info(f"Moyennes mobiles ajoutées pour les périodes: {periods}")
    
    def add_rsi(self, period: int = 14):
        """
        Ajoute l'indicateur RSI (Relative Strength Index)
        
        Args:
            period (int): Période pour le calcul du RSI
        """
        self.data['RSI'] = ta.momentum.rsi(self.data['Close'], window=period)
        logger.info(f"RSI ajouté avec période {period}")
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Ajoute l'indicateur MACD (Moving Average Convergence Divergence)
        
        Args:
            fast (int): Période rapide
            slow (int): Période lente
            signal (int): Période du signal
        """
        # MACD
        macd = ta.trend.MACD(
            self.data['Close'], 
            window_fast=fast, 
            window_slow=slow, 
            window_sign=signal
        )
        
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        self.data['MACD_Histogram'] = macd.macd_diff()
        
        logger.info(f"MACD ajouté (fast={fast}, slow={slow}, signal={signal})")
    
    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2):
        """
        Ajoute les bandes de Bollinger
        
        Args:
            period (int): Période pour la moyenne mobile
            std_dev (float): Nombre d'écarts-types
        """
        bb = ta.volatility.BollingerBands(
            self.data['Close'], 
            window=period, 
            window_dev=std_dev
        )
        
        self.data['BB_Upper'] = bb.bollinger_hband()
        self.data['BB_Middle'] = bb.bollinger_mavg()
        self.data['BB_Lower'] = bb.bollinger_lband()
        self.data['BB_Width'] = bb.bollinger_wband()
        self.data['BB_Position'] = bb.bollinger_pband()
        
        logger.info(f"Bandes de Bollinger ajoutées (période={period}, std={std_dev})")
    
    def add_stochastic(self, k_period: int = 14, d_period: int = 3):
        """
        Ajoute l'oscillateur stochastique
        
        Args:
            k_period (int): Période pour %K
            d_period (int): Période pour %D
        """
        stoch = ta.momentum.StochasticOscillator(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'],
            window=k_period,
            smooth_window=d_period
        )
        
        self.data['Stoch_K'] = stoch.stoch()
        self.data['Stoch_D'] = stoch.stoch_signal()
        
        logger.info(f"Stochastique ajouté (K={k_period}, D={d_period})")
    
    def add_williams_r(self, period: int = 14):
        """
        Ajoute l'indicateur Williams %R
        
        Args:
            period (int): Période pour le calcul
        """
        self.data['Williams_R'] = ta.momentum.williams_r(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            lbp=period
        )
        
        logger.info(f"Williams %R ajouté avec période {period}")
    
    def add_atr(self, period: int = 14):
        """
        Ajoute l'Average True Range (ATR)
        
        Args:
            period (int): Période pour le calcul
        """
        self.data['ATR'] = ta.volatility.average_true_range(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            window=period
        )
        
        logger.info(f"ATR ajouté avec période {period}")
    
    def add_volume_indicators(self):
        """
        Ajoute les indicateurs de volume
        """
        # Volume moyen simple
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        
        # On Balance Volume (OBV)
        self.data['OBV'] = ta.volume.on_balance_volume(
            self.data['Close'], 
            self.data['Volume']
        )
        
        # Volume Price Trend (VPT)
        self.data['VPT'] = ta.volume.volume_price_trend(
            self.data['Close'], 
            self.data['Volume']
        )
        
        # Accumulation/Distribution Line
        self.data['ADL'] = ta.volume.acc_dist_index(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            self.data['Volume']
        )
        
        # Chaikin Money Flow
        self.data['CMF'] = ta.volume.chaikin_money_flow(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            self.data['Volume'],
            window=20
        )
        
        logger.info("Indicateurs de volume ajoutés")
    
    def add_adx(self, period: int = 14):
        """
        Ajoute l'Average Directional Index (ADX)
        
        Args:
            period (int): Période pour le calcul
        """
        adx = ta.trend.ADXIndicator(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            window=period
        )
        
        self.data['ADX'] = adx.adx()
        self.data['ADX_Pos'] = adx.adx_pos()
        self.data['ADX_Neg'] = adx.adx_neg()
        
        logger.info(f"ADX ajouté avec période {period}")
    
    def add_ichimoku(self):
        """
        Ajoute l'indicateur Ichimoku
        """
        ichimoku = ta.trend.IchimokuIndicator(
            self.data['High'], 
            self.data['Low']
        )
        
        self.data['Ichimoku_A'] = ichimoku.ichimoku_a()
        self.data['Ichimoku_B'] = ichimoku.ichimoku_b()
        self.data['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        self.data['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        
        logger.info("Ichimoku ajouté")
    
    def get_signals(self) -> Dict[str, str]:
        """
        Génère des signaux de trading basés sur les indicateurs
        
        Returns:
            dict: Dictionnaire avec les signaux
        """
        signals = {}
        
        # RSI
        if 'RSI' in self.data.columns:
            last_rsi = self.data['RSI'].iloc[-1]
            if last_rsi > 70:
                signals['RSI'] = 'Vente (Survente)'
            elif last_rsi < 30:
                signals['RSI'] = 'Achat (Survente)'
            else:
                signals['RSI'] = 'Neutre'
        
        # MACD
        if 'MACD' in self.data.columns and 'MACD_Signal' in self.data.columns:
            last_macd = self.data['MACD'].iloc[-1]
            last_signal = self.data['MACD_Signal'].iloc[-1]
            prev_macd = self.data['MACD'].iloc[-2]
            prev_signal = self.data['MACD_Signal'].iloc[-2]
            
            if last_macd > last_signal and prev_macd <= prev_signal:
                signals['MACD'] = 'Achat (Croisement haussier)'
            elif last_macd < last_signal and prev_macd >= prev_signal:
                signals['MACD'] = 'Vente (Croisement baissier)'
            else:
                signals['MACD'] = 'Neutre'
        
        # Bandes de Bollinger
        if 'BB_Position' in self.data.columns:
            last_bb_pos = self.data['BB_Position'].iloc[-1]
            if last_bb_pos > 1:
                signals['Bollinger'] = 'Vente (Au-dessus de la bande supérieure)'
            elif last_bb_pos < 0:
                signals['Bollinger'] = 'Achat (En dessous de la bande inférieure)'
            else:
                signals['Bollinger'] = 'Neutre'
        
        # Stochastique
        if 'Stoch_K' in self.data.columns:
            last_stoch_k = self.data['Stoch_K'].iloc[-1]
            if last_stoch_k > 80:
                signals['Stochastique'] = 'Vente (Survente)'
            elif last_stoch_k < 20:
                signals['Stochastique'] = 'Achat (Survente)'
            else:
                signals['Stochastique'] = 'Neutre'
        
        return signals
    
    def plot_indicators(self, symbol: str = "Action"):
        """
        Affiche les graphiques des indicateurs techniques
        
        Args:
            symbol (str): Nom du symbole à afficher
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'Analyse Technique - {symbol}', fontsize=16)
        
        # Prix et moyennes mobiles
        axes[0].plot(self.data.index, self.data['Close'], label='Prix de clôture', linewidth=2)
        if 'SMA_20' in self.data.columns:
            axes[0].plot(self.data.index, self.data['SMA_20'], label='SMA 20', alpha=0.7)
        if 'SMA_50' in self.data.columns:
            axes[0].plot(self.data.index, self.data['SMA_50'], label='SMA 50', alpha=0.7)
        if 'BB_Upper' in self.data.columns:
            axes[0].plot(self.data.index, self.data['BB_Upper'], label='Bande supérieure', alpha=0.5)
            axes[0].plot(self.data.index, self.data['BB_Lower'], label='Bande inférieure', alpha=0.5)
            axes[0].fill_between(self.data.index, self.data['BB_Upper'], self.data['BB_Lower'], alpha=0.1)
        axes[0].set_title('Prix et Moyennes Mobiles')
        axes[0].legend()
        axes[0].grid(True)
        
        # RSI
        if 'RSI' in self.data.columns:
            axes[1].plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7)
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7)
            axes[1].set_title('RSI (Relative Strength Index)')
            axes[1].set_ylabel('RSI')
            axes[1].grid(True)
        
        # MACD
        if 'MACD' in self.data.columns:
            axes[2].plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
            axes[2].plot(self.data.index, self.data['MACD_Signal'], label='Signal', color='red')
            axes[2].bar(self.data.index, self.data['MACD_Histogram'], label='Histogramme', alpha=0.3)
            axes[2].set_title('MACD')
            axes[2].legend()
            axes[2].grid(True)
        
        # Volume
        axes[3].bar(self.data.index, self.data['Volume'], alpha=0.5, label='Volume')
        if 'Volume_SMA' in self.data.columns:
            axes[3].plot(self.data.index, self.data['Volume_SMA'], label='Volume SMA', color='red')
        axes[3].set_title('Volume')
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_summary_stats(self) -> Dict:
        """
        Retourne des statistiques résumées des indicateurs
        
        Returns:
            dict: Statistiques des indicateurs
        """
        stats = {}
        
        if 'RSI' in self.data.columns:
            stats['RSI'] = {
                'Actuel': round(self.data['RSI'].iloc[-1], 2),
                'Moyenne': round(self.data['RSI'].mean(), 2),
                'Min': round(self.data['RSI'].min(), 2),
                'Max': round(self.data['RSI'].max(), 2)
            }
        
        if 'ATR' in self.data.columns:
            stats['ATR'] = {
                'Actuel': round(self.data['ATR'].iloc[-1], 2),
                'Moyenne': round(self.data['ATR'].mean(), 2)
            }
        
        if 'BB_Width' in self.data.columns:
            stats['Bollinger_Width'] = {
                'Actuel': round(self.data['BB_Width'].iloc[-1], 2),
                'Moyenne': round(self.data['BB_Width'].mean(), 2)
            }
        
        return stats

# Exemple d'utilisation
if __name__ == "__main__":
    from data_collector import DataCollector
    
    # Récupérer des données
    collector = DataCollector()
    data = collector.get_stock_data('AAPL', period="6mo")
    
    if data is not None:
        # Créer l'analyseur technique
        ta_analyzer = TechnicalAnalysis(data)
        
        # Ajouter tous les indicateurs
        ta_analyzer.add_all_indicators()
        
        # Obtenir les signaux
        signals = ta_analyzer.get_signals()
        print("Signaux de trading:", signals)
        
        # Statistiques
        stats = ta_analyzer.get_summary_stats()
        print("Statistiques:", stats)
        
        # Afficher les graphiques
        ta_analyzer.plot_indicators('AAPL') 