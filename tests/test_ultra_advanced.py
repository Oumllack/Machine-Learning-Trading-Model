#!/usr/bin/env python3
"""
Test du LSTM Ultra-Avancé avec données étendues et architectures de pointe
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_ultra import UltraAdvancedLSTM
from data_collector import DataCollector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ultra_advanced_lstm(symbol: str = "AAPL"):
    """Test du LSTM ultra-avancé avec données étendues"""
    print("🚀 TEST LSTM ULTRA-AVANCÉ V2.0")
    print(f"Symbole: {symbol}")
    print("=" * 60)
    
    try:
        # Initialisation du prédicteur ultra-avancé
        lstm_ultra = UltraAdvancedLSTM(symbol=symbol)
        
        # Lancement de l'analyse complète
        print("\n🚀 TEST LSTM ULTRA-AVANCÉ")
        print("-" * 30)
        
        metrics = lstm_ultra.run_ultra_analysis(sequence_length=20)
        
        print(f"\n📈 Métriques LSTM Ultra-Avancé V2.0:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.4f}")
        print(f"  Direction_Accuracy: {metrics['Direction_Accuracy']:.4f}")
        print(f"  Profit_Factor: {metrics['Profit_Factor']:.4f}")
        
        print(f"\n🎯 ANALYSE DES RÉSULTATS")
        print("-" * 30)
        
        # Critères de performance
        criteria_met = 0
        total_criteria = 4
        
        if metrics['R2'] > 0.3:
            print("✅ R² > 0.3 - Bonne capacité prédictive")
            criteria_met += 1
        else:
            print(f"⚠️ R² = {metrics['R2']:.4f} - Capacité prédictive à améliorer")
        
        if metrics['Direction_Accuracy'] > 55:
            print("✅ Direction Accuracy > 55% - Bonne prédiction de direction")
            criteria_met += 1
        else:
            print(f"⚠️ Direction Accuracy = {metrics['Direction_Accuracy']:.2f}% - Prédiction de direction à améliorer")
        
        if metrics['MAPE'] < 50:
            print("✅ MAPE < 50% - Précision acceptable")
            criteria_met += 1
        else:
            print(f"⚠️ MAPE = {metrics['MAPE']:.2f}% - Précision à améliorer")
        
        if metrics['Profit_Factor'] > 1.2:
            print("✅ Profit Factor > 1.2 - Potentiel de profit")
            criteria_met += 1
        else:
            print(f"⚠️ Profit Factor = {metrics['Profit_Factor']:.4f} - Potentiel de profit à améliorer")
        
        print(f"\n🎉 PERFORMANCE GLOBALE: {criteria_met}/{total_criteria} critères réussis")
        
        if criteria_met >= 3:
            print("🌟 EXCELLENT - Le modèle est prêt pour le trading!")
        elif criteria_met >= 2:
            print("👍 BON - Le modèle montre des signes prometteurs")
        elif criteria_met >= 1:
            print("⚠️ MOYEN - Le modèle nécessite des améliorations")
        else:
            print("❌ INSUFFISANT - Le modèle nécessite une refonte complète")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Erreur LSTM Ultra-Avancé: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test sur Apple
    test_ultra_advanced_lstm("AAPL") 