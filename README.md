# 🤖 Système de Trading Automatique Avancé

Un système complet de trading automatique en Python intégrant la récupération de données boursières, l'analyse technique, la prédiction LSTM et des bots de trading intelligents.

## 📊 Résultats des Simulations de Trading Réelles

### 🎯 Comparaison Conservateur vs Agressif

Nous avons exécuté des simulations réelles sur 30 jours pour trois actions majeures :

#### 📈 AAPL (Apple Inc.)
- **Bot Conservateur**: 0.00% (0 trades) - Aucune opportunité détectée
- **Bot Agressif**: -1.69% (9 trades) - Taux de réussite: 22.22%
  - Gain moyen: 46.06€
  - Perte moyenne: -25.24€
  - Capital final: 4,915.44€

#### 💻 MSFT (Microsoft Corporation)
- **Bot Conservateur**: 0.00% (0 trades) - Aucune opportunité détectée
- **Bot Agressif**: +0.02% (1 trade) - Taux de réussite: 100%
  - Gain moyen: 1.22€
  - Perte moyenne: 0.00€
  - Capital final: 5,001.22€

#### 🚗 TSLA (Tesla Inc.)
- **Bot Conservateur**: 0.00% (0 trades) - Aucune opportunité détectée
- **Bot Agressif**: -0.20% (1 trade) - Taux de réussite: 0%
  - Gain moyen: 0.00€
  - Perte moyenne: -9.87€
  - Capital final: 4,990.13€

### 📊 Graphiques Générés

Les simulations ont produit des graphiques détaillés pour chaque action :

- **Prix et Trades**: Visualisation des points d'entrée et de sortie
- **Évolution du Portefeuille**: Suivi de la valeur du capital
- **Indicateurs Techniques**: RSI, MACD avec seuils adaptatifs
- **Distribution des P&L**: Analyse des gains et pertes
- **Résumé des Performances**: Métriques détaillées

### 🔍 Observations Clés

1. **Bot Conservateur**: Très sélectif, aucun trade effectué sur la période testée
2. **Bot Agressif**: Plus actif avec des paramètres permissifs
3. **Gestion du Risque**: Stop loss et take profit automatiques
4. **Analyse Technique**: Utilisation de RSI, MACD et moyennes mobiles

## 🚀 Fonctionnalités

### 📈 Collecte de Données
- **Yahoo Finance**: Récupération en temps réel
- **Indicateurs Techniques**: RSI, MACD, Bandes de Bollinger, Moyennes mobiles
- **Données Historiques**: Jusqu'à 2 ans de données

### 🧠 Analyse Technique
- **RSI (Relative Strength Index)**: Détection de surachat/survente
- **MACD**: Convergence/divergence des moyennes mobiles
- **Bandes de Bollinger**: Volatilité et niveaux de support/résistance
- **Moyennes Mobiles**: Tendances court et long terme

### 🤖 Bots de Trading

#### Bot Conservateur
- Seuil de confiance élevé (0.4)
- Stop loss: 3%
- Take profit: 8%
- Position max: 20% du capital
- Risque par trade: 3%

#### Bot Agressif
- Seuil de confiance bas (0.15)
- Stop loss: 1.5%
- Take profit: 4%
- Position max: 30% du capital
- Risque par trade: 5%

### 🧠 Prédiction LSTM
- **Modèle Ultra-Avancé**: Architecture LSTM complexe
- **Features Multiples**: Prix, volume, indicateurs techniques
- **Prédiction Multi-Horizon**: 1, 5, 10 jours
- **Backtesting**: Validation sur données historiques

## 📁 Structure du Projet

```
Share price prediction/
├── src/
│   ├── core/                 # Modules principaux
│   │   ├── data_collector.py
│   │   ├── technical_analysis.py
│   │   ├── trading_bot_simple.py
│   │   └── lstm_ultra.py
│   ├── demos/               # Scripts de démonstration
│   │   ├── demo_trading_final.py
│   │   ├── demo_trading_aggressive.py
│   │   └── demo_trading_complete.py
│   ├── config/              # Configuration
│   ├── utils/               # Utilitaires
│   └── main.py              # Interface principale
├── images/                  # Graphiques générés
├── logs/                    # Logs de trading
├── tests/                   # Tests unitaires
├── docs/                    # Documentation
├── requirements.txt         # Dépendances
└── README.md               # Ce fichier
```

## 🛠️ Installation

### Prérequis
```bash
Python 3.8+
pip
```

### Installation des Dépendances
```bash
pip install -r requirements.txt
```

### Dépendances Principales
- `yfinance`: Données boursières
- `pandas`: Manipulation de données
- `numpy`: Calculs numériques
- `matplotlib`: Graphiques
- `scikit-learn`: Machine Learning
- `tensorflow`: Modèles LSTM (optionnel)

## 🚀 Utilisation

### Démonstration Complète
```bash
cd src/demos
python demo_trading_final.py
```

### Simulation Conservateur vs Agressif
```bash
python demo_trading_final.py
```

### Simulation sur une Action Spécifique
```bash
python demo_trading_final.py single AAPL conservative
python demo_trading_final.py single MSFT aggressive
```

### Interface Principale
```bash
cd src
python main.py
```

## 📊 Exemples de Commandes

### Analyse Technique
```python
from src.core.data_collector import DataCollector
from src.core.technical_analysis import TechnicalAnalysis

# Récupérer les données
collector = DataCollector()
data = collector.get_stock_data('AAPL', period='1y')

# Analyse technique
analyzer = TechnicalAnalysis(data)
analyzer.add_all_indicators()
signals = analyzer.get_signals()
```

### Bot de Trading
```python
from src.core.trading_bot_simple import SimpleTradingBot

# Créer un bot
bot = SimpleTradingBot(
    symbol='AAPL',
    initial_capital=10000,
    risk_per_trade=0.03,
    max_position_size=0.2
)

# Lancer une session
bot.run_trading_session(days=30)
metrics = bot.get_performance_metrics()
```

## 📈 Résultats des Tests

### Performance des Bots

| Action | Bot Conservateur | Bot Agressif | Meilleur |
|--------|------------------|--------------|----------|
| AAPL   | 0.00% (0 trades) | -1.69% (9 trades) | Conservateur |
| MSFT   | 0.00% (0 trades) | +0.02% (1 trade) | Agressif |
| TSLA   | 0.00% (0 trades) | -0.20% (1 trade) | Conservateur |

### Observations
- **Bot Conservateur**: Évite les pertes mais manque d'opportunités
- **Bot Agressif**: Plus d'activité mais risque de pertes
- **Gestion du Risque**: Cruciale pour la performance

## 🔧 Configuration

### Paramètres du Bot Conservateur
```python
confidence_threshold = 0.4
stop_loss_pct = 0.03
take_profit_pct = 0.08
max_position_size = 0.2
risk_per_trade = 0.03
```

### Paramètres du Bot Agressif
```python
confidence_threshold = 0.15
stop_loss_pct = 0.015
take_profit_pct = 0.04
max_position_size = 0.3
risk_per_trade = 0.05
```

## 📊 Graphiques Disponibles

### Graphiques par Action
- `trading_simulation_AAPL_conservative.png`
- `trading_simulation_AAPL_aggressive.png`
- `trading_simulation_MSFT_conservative.png`
- `trading_simulation_MSFT_aggressive.png`
- `trading_simulation_TSLA_conservative.png`
- `trading_simulation_TSLA_aggressive.png`

### Graphiques de Comparaison
- `trading_comparison_final.png`: Comparaison complète

## 🧪 Tests

### Tests Unitaires
```bash
cd tests
python -m pytest
```

### Tests de Performance
```bash
python test_trading_bot.py
python test_lstm_predictor.py
```

## 📝 Logs et Monitoring

### Logs de Trading
- Fichiers dans `logs/`
- Format: `trading_system_YYYYMMDD.log`
- Détails des trades et performances

### Métriques de Performance
- Rendement total
- Nombre de trades
- Taux de réussite
- Gain/perte moyen
- Ratio de Sharpe

## 🔒 Gestion du Risque

### Stop Loss Automatique
- Protection contre les pertes importantes
- Seuils configurables par bot
- Exécution automatique

### Take Profit
- Sécurisation des gains
- Niveaux adaptatifs
- Optimisation des rendements

### Position Sizing
- Limitation de l'exposition
- Calcul basé sur le risque
- Diversification automatique

## 🚀 Améliorations Futures

### Fonctionnalités Planifiées
- [ ] Interface web
- [ ] Trading en temps réel
- [ ] Plus d'indicateurs techniques
- [ ] Optimisation des paramètres
- [ ] Backtesting avancé
- [ ] Gestion multi-actifs

### Optimisations Techniques
- [ ] Parallélisation des calculs
- [ ] Cache des données
- [ ] Optimisation mémoire
- [ ] API REST

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! Veuillez :

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📞 Support

Pour toute question ou problème :
- Ouvrir une issue sur GitHub
- Consulter la documentation dans `docs/`
- Vérifier les logs dans `logs/`

---

**⚠️ Avertissement**: Ce système est destiné à des fins éducatives et de recherche. Le trading comporte des risques de perte. Utilisez à vos propres risques. 