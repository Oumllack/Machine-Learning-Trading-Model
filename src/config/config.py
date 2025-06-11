"""
Fichier de configuration pour l'analyse boursière
Personnalisez les paramètres selon vos besoins
"""

# Configuration générale
GENERAL_CONFIG = {
    'default_period': '2y',  # Période par défaut pour les données
    'default_sequence_length': 60,  # Longueur de séquence LSTM par défaut
    'save_plots': True,  # Sauvegarder les graphiques
    'save_data': True,  # Sauvegarder les données
    'log_level': 'INFO',  # Niveau de logging (DEBUG, INFO, WARNING, ERROR)
}

# Configuration des indicateurs techniques
TECHNICAL_INDICATORS = {
    'rsi': {
        'period': 14,
        'overbought': 70,
        'oversold': 30
    },
    'macd': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    },
    'bollinger_bands': {
        'period': 20,
        'std_dev': 2
    },
    'moving_averages': {
        'periods': [5, 10, 20, 50, 200]
    },
    'stochastic': {
        'k_period': 14,
        'd_period': 3
    },
    'williams_r': {
        'period': 14
    },
    'atr': {
        'period': 14
    },
    'adx': {
        'period': 14
    }
}

# Configuration LSTM
LSTM_CONFIG = {
    'architecture': {
        'lstm_units': [50, 50],  # Unités par couche LSTM
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'dense_units': 25
    },
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'patience': 15,  # Early stopping
        'validation_split': 0.1,
        'test_split': 0.2
    },
    'data_preparation': {
        'sequence_length': 60,
        'features': ['Open', 'High', 'Low', 'Close', 'Volume'],
        'target_column': 'Close',
        'normalization': 'minmax'  # 'minmax' ou 'standard'
    }
}

# Configuration des symboles populaires
POPULAR_SYMBOLS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
    'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V'],
    'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'MRK', 'ABT', 'DHR'],
    'energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC'],
    'oil': ['BZ=F', 'CL=F'],  # Brent et WTI
    'indices': ['^GSPC', '^IXIC', '^DJI', '^VIX']  # S&P 500, NASDAQ, Dow Jones, VIX
}

# Configuration des alertes
ALERTS_CONFIG = {
    'rsi_alerts': {
        'enabled': True,
        'overbought_threshold': 70,
        'oversold_threshold': 30
    },
    'price_alerts': {
        'enabled': True,
        'percentage_change': 5.0  # Alerte si changement > 5%
    },
    'volume_alerts': {
        'enabled': True,
        'volume_multiplier': 2.0  # Alerte si volume > 2x la moyenne
    }
}

# Configuration des rapports
REPORTS_CONFIG = {
    'include_technical_analysis': True,
    'include_lstm_predictions': True,
    'include_company_info': True,
    'include_charts': True,
    'save_format': 'txt',  # 'txt', 'html', 'pdf'
    'include_future_predictions': True,
    'prediction_days': 30
}

# Configuration des visualisations
VISUALIZATION_CONFIG = {
    'style': 'seaborn-v0_8',
    'figure_size': (15, 10),
    'dpi': 100,
    'save_format': 'png',
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e'
    }
}

# Configuration des données
DATA_CONFIG = {
    'cache_data': True,
    'cache_duration': 3600,  # 1 heure en secondes
    'max_retries': 3,
    'timeout': 30,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Configuration des métriques de performance
PERFORMANCE_METRICS = {
    'lstm_thresholds': {
        'excellent_mape': 5.0,   # MAPE < 5% = Excellent
        'good_mape': 10.0,       # MAPE < 10% = Bon
        'acceptable_mape': 20.0, # MAPE < 20% = Acceptable
        'excellent_r2': 0.8,     # R² > 0.8 = Excellent
        'good_r2': 0.6,          # R² > 0.6 = Bon
        'acceptable_r2': 0.4     # R² > 0.4 = Acceptable
    }
}

# Configuration des signaux de trading
TRADING_SIGNALS = {
    'rsi': {
        'buy_signal': 'RSI < 30',
        'sell_signal': 'RSI > 70',
        'neutral': '30 <= RSI <= 70'
    },
    'macd': {
        'buy_signal': 'MACD > Signal (croisement haussier)',
        'sell_signal': 'MACD < Signal (croisement baissier)',
        'neutral': 'Pas de croisement'
    },
    'bollinger': {
        'buy_signal': 'Prix < Bande inférieure',
        'sell_signal': 'Prix > Bande supérieure',
        'neutral': 'Prix entre les bandes'
    },
    'moving_averages': {
        'buy_signal': 'Prix > SMA 20 > SMA 50',
        'sell_signal': 'Prix < SMA 20 < SMA 50',
        'neutral': 'Croisements mixtes'
    }
}

# Configuration des timeframes
TIMEFRAMES = {
    'intraday': ['1m', '5m', '15m', '30m', '1h'],
    'daily': ['1d', '5d', '1wk', '1mo'],
    'long_term': ['3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
}

# Configuration des exports
EXPORT_CONFIG = {
    'formats': ['csv', 'parquet', 'json', 'excel'],
    'default_format': 'csv',
    'include_indicators': True,
    'include_predictions': True,
    'timestamp_format': '%Y%m%d_%H%M%S'
}

# Configuration des modèles
MODEL_CONFIG = {
    'save_models': True,
    'model_format': 'h5',
    'load_pretrained': False,
    'pretrained_path': None,
    'model_versioning': True
}

# Configuration des tests
TEST_CONFIG = {
    'quick_test_epochs': 5,
    'quick_test_sequence_length': 30,
    'quick_test_lstm_units': [20, 10],
    'full_test_epochs': 50,
    'test_symbols': ['AAPL', 'MSFT', 'TSLA']
}

# =============================================================================
# RISK MANAGEMENT PARAMETERS
# =============================================================================

# Position sizing
RISK_PER_TRADE = 0.02  # 2% risk per trade
MAX_POSITION_SIZE = 0.1  # 10% maximum position size
MIN_POSITION_SIZE = 0.01  # 1% minimum position size

# Stop-loss and take-profit
STOP_LOSS_PCT = 0.05  # 5% stop loss
TAKE_PROFIT_PCT = 0.15  # 15% take profit
TRAILING_STOP_PCT = 0.03  # 3% trailing stop

# Risk thresholds
MAX_DAILY_LOSS = 0.05  # 5% maximum daily loss
MAX_PORTFOLIO_DRAWDOWN = 0.20  # 20% maximum portfolio drawdown

# =============================================================================
# TECHNICAL ANALYSIS PARAMETERS
# =============================================================================

# Moving averages
SMA_PERIODS = [5, 10, 20, 50, 200]
EMA_PERIODS = [12, 26]

# RSI
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Stochastic
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERBOUGHT = 80
STOCH_OVERSOLD = 20

# Williams %R
WILLIAMS_R_PERIOD = 14
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2

# ATR
ATR_PERIOD = 14

# ADX
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# =============================================================================
# TRADING ALGORITHM PARAMETERS
# =============================================================================

# Signal generation
CONFIDENCE_THRESHOLD = 0.6  # 60% confidence required for trade
SIGNAL_WEIGHTS = {
    'RSI': 0.2,
    'MACD': 0.2,
    'Stochastic': 0.15,
    'Williams_R': 0.15,
    'Bollinger_Bands': 0.1,
    'Moving_Averages': 0.1,
    'Volume': 0.1
}

# Trend analysis
TREND_STRENGTH_THRESHOLD = 0.7
TREND_CONFIRMATION_PERIODS = 3

# Volume analysis
VOLUME_THRESHOLD = 1.5  # 1.5x average volume
VOLUME_CONFIRMATION = True

# =============================================================================
# PORTFOLIO MANAGEMENT
# =============================================================================

# Portfolio settings
INITIAL_CAPITAL = 10000
MAX_STOCKS = 10
REBALANCE_FREQUENCY = 30  # days

# Diversification
SECTOR_LIMIT = 0.3  # 30% maximum per sector
STOCK_LIMIT = 0.2  # 20% maximum per stock

# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

# Performance metrics
BENCHMARK = 'SPY'  # S&P 500 ETF
RISK_FREE_RATE = 0.02  # 2% risk-free rate

# Monitoring thresholds
PERFORMANCE_ALERT_THRESHOLD = -0.05  # -5% performance alert
DRAWDOWN_ALERT_THRESHOLD = 0.10  # 10% drawdown alert

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================

# Data settings
DATA_SOURCE = 'yahoo'
DEFAULT_PERIOD = '2y'
UPDATE_FREQUENCY = '1d'

# Trading settings
TRADING_MODE = 'paper'  # 'paper' or 'live'
COMMISSION_RATE = 0.001  # 0.1% commission
SLIPPAGE = 0.0005  # 0.05% slippage

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_system.log'

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Machine learning (future implementation)
ML_ENABLED = False
ML_MODEL_PATH = 'models/'
ML_RETRAIN_FREQUENCY = 30  # days

# Alternative data (future implementation)
ALTERNATIVE_DATA_ENABLED = False
NEWS_SENTIMENT_WEIGHT = 0.1
SOCIAL_MEDIA_WEIGHT = 0.05

# Market regime detection
REGIME_DETECTION_ENABLED = True
REGIME_UPDATE_FREQUENCY = 5  # days

# =============================================================================
# OPTIMIZATION SETTINGS
# =============================================================================

# Parameter optimization
OPTIMIZATION_ENABLED = True
OPTIMIZATION_FREQUENCY = 90  # days
OPTIMIZATION_METHOD = 'genetic'  # 'genetic', 'bayesian', 'grid'

# Backtesting
BACKTEST_START_DATE = '2022-01-01'
BACKTEST_END_DATE = '2024-12-01'
WALK_FORWARD_WINDOW = 252  # days

# =============================================================================
# NOTIFICATION SETTINGS
# =============================================================================

# Email notifications
EMAIL_ENABLED = False
EMAIL_SMTP_SERVER = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_FROM = 'trading@example.com'
EMAIL_TO = 'admin@example.com'

# Slack notifications
SLACK_ENABLED = False
SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/...'

# =============================================================================
# ENVIRONMENT-SPECIFIC SETTINGS
# =============================================================================

# Development environment
if __name__ == "__main__":
    # Override settings for development
    TRADING_MODE = 'paper'
    LOG_LEVEL = 'DEBUG'
    OPTIMIZATION_ENABLED = False

def get_config():
    """
    Retourne la configuration complète
    """
    return {
        'general': GENERAL_CONFIG,
        'technical_indicators': TECHNICAL_INDICATORS,
        'lstm': LSTM_CONFIG,
        'popular_symbols': POPULAR_SYMBOLS,
        'alerts': ALERTS_CONFIG,
        'reports': REPORTS_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'data': DATA_CONFIG,
        'performance_metrics': PERFORMANCE_METRICS,
        'trading_signals': TRADING_SIGNALS,
        'timeframes': TIMEFRAMES,
        'export': EXPORT_CONFIG,
        'model': MODEL_CONFIG,
        'test': TEST_CONFIG
    }

def update_config(new_config):
    """
    Met à jour la configuration avec de nouveaux paramètres
    
    Args:
        new_config (dict): Nouvelle configuration
    """
    global GENERAL_CONFIG, TECHNICAL_INDICATORS, LSTM_CONFIG
    
    if 'general' in new_config:
        GENERAL_CONFIG.update(new_config['general'])
    
    if 'technical_indicators' in new_config:
        TECHNICAL_INDICATORS.update(new_config['technical_indicators'])
    
    if 'lstm' in new_config:
        LSTM_CONFIG.update(new_config['lstm'])

# Exemple d'utilisation
if __name__ == "__main__":
    config = get_config()
    print("Configuration chargée avec succès")
    print(f"Période par défaut: {config['general']['default_period']}")
    print(f"Unités LSTM: {config['lstm']['architecture']['lstm_units']}")
    print(f"Symboles tech populaires: {config['popular_symbols']['tech']}") 