"""
Module de récupération des données boursières via Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Classe pour récupérer les données boursières via Yahoo Finance
    """
    
    def __init__(self):
        self.data = None
    
    def get_stock_data(self, symbol, start_date=None, end_date=None, period="2y"):
        """
        Récupère les données historiques d'une action
        
        Args:
            symbol (str): Symbole de l'action (ex: 'AAPL', 'MSFT', 'TSLA')
            start_date (str): Date de début au format 'YYYY-MM-DD'
            end_date (str): Date de fin au format 'YYYY-MM-DD'
            period (str): Période par défaut si pas de dates spécifiées
            
        Returns:
            pd.DataFrame: DataFrame avec les données historiques
        """
        try:
            logger.info(f"Récupération des données pour {symbol}")
            
            # Créer l'objet Ticker
            ticker = yf.Ticker(symbol)
            
            # Récupérer les données historiques
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period=period)
            
            # Ajouter des informations sur le ticker
            data['Symbol'] = symbol
            data['Company'] = ticker.info.get('longName', symbol)
            
            self.data = data
            logger.info(f"Données récupérées avec succès: {len(data)} enregistrements")
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour {symbol}: {str(e)}")
            return None
    
    def get_oil_data(self, oil_type="BZ=F"):
        """
        Récupère les données du pétrole (Brent ou WTI)
        
        Args:
            oil_type (str): Type de pétrole ('BZ=F' pour Brent, 'CL=F' pour WTI)
            
        Returns:
            pd.DataFrame: DataFrame avec les données du pétrole
        """
        oil_names = {
            'BZ=F': 'Brent Crude Oil',
            'CL=F': 'WTI Crude Oil'
        }
        
        return self.get_stock_data(oil_type, period="2y")
    
    def get_multiple_stocks(self, symbols, start_date=None, end_date=None):
        """
        Récupère les données pour plusieurs actions
        
        Args:
            symbols (list): Liste des symboles d'actions
            start_date (str): Date de début
            end_date (str): Date de fin
            
        Returns:
            dict: Dictionnaire avec les données de chaque action
        """
        all_data = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, start_date, end_date)
            if data is not None:
                all_data[symbol] = data
        
        return all_data
    
    def get_company_info(self, symbol):
        """
        Récupère les informations sur une entreprise
        
        Args:
            symbol (str): Symbole de l'action
            
        Returns:
            dict: Informations sur l'entreprise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Informations essentielles
            essential_info = {
                'Nom': info.get('longName', 'N/A'),
                'Secteur': info.get('sector', 'N/A'),
                'Industrie': info.get('industry', 'N/A'),
                'Pays': info.get('country', 'N/A'),
                'Capitalisation': info.get('marketCap', 'N/A'),
                'Volume moyen': info.get('averageVolume', 'N/A'),
                'Dividende': info.get('dividendYield', 'N/A'),
                'PER': info.get('trailingPE', 'N/A'),
                'Prix actuel': info.get('currentPrice', 'N/A')
            }
            
            return essential_info
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos pour {symbol}: {str(e)}")
            return None
    
    def save_data(self, filename=None, format='csv'):
        """
        Sauvegarde les données récupérées
        
        Args:
            filename (str): Nom du fichier
            format (str): Format de sauvegarde ('csv' ou 'parquet')
        """
        if self.data is None:
            logger.warning("Aucune donnée à sauvegarder")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_data_{timestamp}"
        
        try:
            if format == 'csv':
                self.data.to_csv(f"{filename}.csv")
            elif format == 'parquet':
                self.data.to_parquet(f"{filename}.parquet")
            
            logger.info(f"Données sauvegardées dans {filename}.{format}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Test du collecteur de données
    collector = DataCollector()
    
    # Récupérer les données d'Apple
    apple_data = collector.get_stock_data('AAPL', period="1y")
    print(f"Données Apple: {len(apple_data)} enregistrements")
    
    # Récupérer les données du pétrole Brent
    oil_data = collector.get_oil_data('BZ=F')
    print(f"Données Brent: {len(oil_data)} enregistrements")
    
    # Informations sur Apple
    apple_info = collector.get_company_info('AAPL')
    print("Informations Apple:", apple_info) 