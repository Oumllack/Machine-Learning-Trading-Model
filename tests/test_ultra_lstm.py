#!/usr/bin/env python3
"""
Script de test pour le LSTM ultra-avancé
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector import DataCollector
from lstm_ultra import UltraLSTMPredictor
import pandas as pd
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ultra_lstm(symbol: str = "AAPL"):
    """
    Test du LSTM ultra-avancé
    """
    print(f"🚀 TEST LSTM ULTRA-AVANCÉ")
    print(f"Symbole: {symbol}")
    print("=" * 60)
    
    # 1. Récupération des données
    print("📊 Récupération des données...")
    collector = DataCollector()
    data = collector.get_stock_data(symbol, period="3y")  # Plus de données historiques
    
    if data is None or len(data) < 200:
        print("❌ Données insuffisantes")
        return
    
    print(f"✅ Données récupérées: {len(data)} enregistrements")
    
    # 2. Test LSTM Ultra-Avancé
    print("\n🚀 TEST LSTM ULTRA-AVANCÉ")
    print("-" * 30)
    
    try:
        # Préparation des données
        lstm_ultra = UltraLSTMPredictor(data)
        X_train, y_train, X_val, y_val, X_test, y_test = lstm_ultra.prepare_ultra_data(
            sequence_length=15,
            test_size=0.08,
            validation_size=0.12,
            use_ultra_features=True
        )
        
        # Entraînement de l'ensemble
        lstm_ultra.train_ensemble(
            X_train, y_train, X_val, y_val,
            epochs=50,
            batch_size=16
        )
        
        # Évaluation
        metrics_ultra = lstm_ultra.evaluate_ensemble(X_test, y_test)
        
        print("\n📈 Métriques LSTM Ultra-Avancé:")
        for metric, value in metrics_ultra.items():
            print(f"  {metric}: {value:.4f}")
            
        # Analyse des résultats
        print("\n🎯 ANALYSE DES RÉSULTATS")
        print("-" * 30)
        
        if metrics_ultra['R2'] > 0:
            print(f"✅ R² positif: {metrics_ultra['R2']:.4f} - Le modèle est prédictif!")
        else:
            print(f"⚠️ R² négatif: {metrics_ultra['R2']:.4f} - Le modèle nécessite des améliorations")
        
        if metrics_ultra['Direction_Accuracy'] > 60:
            print(f"✅ Direction Accuracy excellente: {metrics_ultra['Direction_Accuracy']:.1f}%")
        elif metrics_ultra['Direction_Accuracy'] > 50:
            print(f"✅ Direction Accuracy bonne: {metrics_ultra['Direction_Accuracy']:.1f}%")
        else:
            print(f"⚠️ Direction Accuracy faible: {metrics_ultra['Direction_Accuracy']:.1f}%")
        
        if metrics_ultra['MAPE'] < 5:
            print(f"✅ MAPE excellente: {metrics_ultra['MAPE']:.2f}%")
        elif metrics_ultra['MAPE'] < 10:
            print(f"✅ MAPE bonne: {metrics_ultra['MAPE']:.2f}%")
        else:
            print(f"⚠️ MAPE élevée: {metrics_ultra['MAPE']:.2f}%")
        
        if metrics_ultra['Profit_Factor'] > 1.5:
            print(f"✅ Profit Factor excellent: {metrics_ultra['Profit_Factor']:.2f}")
        elif metrics_ultra['Profit_Factor'] > 1.0:
            print(f"✅ Profit Factor positif: {metrics_ultra['Profit_Factor']:.2f}")
        else:
            print(f"⚠️ Profit Factor faible: {metrics_ultra['Profit_Factor']:.2f}")
        
        print(f"\n🎉 PERFORMANCE GLOBALE: {len([m for m in [metrics_ultra['R2'] > 0, metrics_ultra['Direction_Accuracy'] > 50, metrics_ultra['MAPE'] < 10, metrics_ultra['Profit_Factor'] > 1.0] if m])}/4 critères réussis")
            
    except Exception as e:
        print(f"❌ Erreur LSTM Ultra-Avancé: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Test avec Apple
    test_ultra_lstm("AAPL") 