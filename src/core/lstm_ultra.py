"""
Module de prédiction LSTM ultra-avancé avec attention, stacking, features enrichies et pipeline complet.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate, GlobalAveragePooling1D, Bidirectional, Conv1D, MaxPooling1D, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import logging
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(1,), initializer='zeros', trainable=True)
        
    def call(self, inputs):
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return output, a

class UltraAdvancedLSTM:
    def __init__(self, symbol: str, target_column: str = 'Close'):
        self.symbol = symbol
        self.target_column = target_column
        self.data = None
        self.scaler = RobustScaler()
        self.models = []
        
    def fetch_extended_data(self, period: str = "5y") -> pd.DataFrame:
        """Récupère plus de données historiques"""
        logger.info(f"Récupération de données étendues pour {self.symbol}")
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=period, interval="1d")
        
        # Ajouter des données supplémentaires
        if len(data) > 0:
            # Données macro-économiques (simulées pour l'exemple)
            data['VIX'] = np.random.normal(20, 5, len(data))  # Volatilité
            data['Interest_Rate'] = np.random.normal(2, 0.5, len(data))  # Taux d'intérêt
            data['GDP_Growth'] = np.random.normal(2, 1, len(data))  # Croissance PIB
            
            # Données sectorielles
            data['Sector_Performance'] = np.random.normal(0, 0.02, len(data))
            data['Market_Sentiment'] = np.random.normal(0.5, 0.2, len(data))
            
        logger.info(f"Données récupérées: {len(data)} enregistrements")
        return data
    
    def create_ultra_features(self) -> pd.DataFrame:
        """Crée des features ultra-avancées"""
        logger.info("Création de features ultra-avancées...")
        df = self.data.copy()
        
        # Features de base
        df['Returns'] = df[self.target_column].pct_change()
        df['Log_Returns'] = np.log(df[self.target_column] / df[self.target_column].shift(1))
        
        # Moyennes mobiles avancées
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{window}'] = df[self.target_column].rolling(window=window).mean()
            df[f'EMA_{window}'] = df[self.target_column].ewm(span=window).mean()
            df[f'Price_SMA_{window}_Ratio'] = df[self.target_column] / df[f'SMA_{window}']
            df[f'Price_EMA_{window}_Ratio'] = df[self.target_column] / df[f'EMA_{window}']
        
        # Volatilité
        for window in [5, 10, 20, 50]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
            df[f'Realized_Volatility_{window}'] = df['Log_Returns'].rolling(window=window).std()
        
        # Momentum
        for period in [5, 10, 20, 50]:
            df[f'Momentum_{period}'] = df[self.target_column] / df[self.target_column].shift(period) - 1
            df[f'ROC_{period}'] = (df[self.target_column] - df[self.target_column].shift(period)) / df[self.target_column].shift(period)
        
        # RSI avec différentes périodes
        for period in [7, 14, 21]:
            delta = df[self.target_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df[self.target_column].ewm(span=12).mean()
        exp2 = df[self.target_column].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        for window in [20, 50]:
            sma = df[self.target_column].rolling(window=window).mean()
            std = df[self.target_column].rolling(window=window).std()
            df[f'BB_Upper_{window}'] = sma + (std * 2)
            df[f'BB_Lower_{window}'] = sma - (std * 2)
            df[f'BB_Width_{window}'] = df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}']
            df[f'BB_Position_{window}'] = (df[self.target_column] - df[f'BB_Lower_{window}']) / df[f'BB_Width_{window}']
        
        # Stochastic Oscillator
        for k_period in [14, 21]:
            lowest_low = df['Low'].rolling(window=k_period).min()
            highest_high = df['High'].rolling(window=k_period).max()
            df[f'Stoch_K_{k_period}'] = 100 * (df[self.target_column] - lowest_low) / (highest_high - lowest_low)
            df[f'Stoch_D_{k_period}'] = df[f'Stoch_K_{k_period}'].rolling(window=3).mean()
        
        # Williams %R
        for period in [14, 21]:
            highest_high = df['High'].rolling(window=period).max()
            lowest_low = df['Low'].rolling(window=period).min()
            df[f'Williams_R_{period}'] = -100 * (highest_high - df[self.target_column]) / (highest_high - lowest_low)
        
        # ATR (Average True Range)
        for period in [14, 21]:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df[self.target_column].shift())
            low_close = np.abs(df['Low'] - df[self.target_column].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df[f'ATR_{period}'] = true_range.rolling(window=period).mean()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Price_Volume_Trend'] = (df['Returns'] * df['Volume']).cumsum()
        
        # Fibonacci retracements
        for period in [20, 50]:
            high = df[self.target_column].rolling(window=period).max()
            low = df[self.target_column].rolling(window=period).min()
            diff = high - low
            df[f'Fib_23.6_{period}'] = high - 0.236 * diff
            df[f'Fib_38.2_{period}'] = high - 0.382 * diff
            df[f'Fib_50.0_{period}'] = high - 0.5 * diff
            df[f'Fib_61.8_{period}'] = high - 0.618 * diff
        
        # Support and Resistance
        for period in [20, 50]:
            df[f'Support_{period}'] = df[self.target_column].rolling(window=period).min()
            df[f'Resistance_{period}'] = df[self.target_column].rolling(window=period).max()
            df[f'Support_Distance_{period}'] = (df[self.target_column] - df[f'Support_{period}']) / df[self.target_column]
            df[f'Resistance_Distance_{period}'] = (df[f'Resistance_{period}'] - df[self.target_column]) / df[self.target_column]
        
        # Trend indicators
        for period in [5, 10, 20]:
            df[f'Trend_{period}'] = np.where(df[f'SMA_{period}'] > df[f'SMA_{period}'].shift(1), 1, -1)
            df[f'Trend_Strength_{period}'] = abs(df[f'SMA_{period}'] - df[f'SMA_{period}'].shift(1)) / df[f'SMA_{period}'].shift(1)
        
        # Seasonality
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # Cyclical encoding
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Price_Lag_{lag}'] = df[self.target_column].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Price_Mean_{window}'] = df[self.target_column].rolling(window=window).mean()
            df[f'Price_Std_{window}'] = df[self.target_column].rolling(window=window).std()
            df[f'Price_Skew_{window}'] = df[self.target_column].rolling(window=window).skew()
            df[f'Price_Kurt_{window}'] = df[self.target_column].rolling(window=window).kurt()
        
        # Cross-sectional features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df[self.target_column] / df['Open']
        df['Body_Size'] = abs(df[self.target_column] - df['Open'])
        df['Upper_Shadow'] = df['High'] - np.maximum(df[self.target_column], df['Open'])
        df['Lower_Shadow'] = np.minimum(df[self.target_column], df['Open']) - df['Low']
        
        # Market regime features
        df['Market_Regime'] = np.where(df['Volatility_20'] > df['Volatility_20'].rolling(window=252).mean(), 'High_Vol', 'Low_Vol')
        df['Trend_Regime'] = np.where(df['SMA_50'] > df['SMA_200'], 'Uptrend', 'Downtrend')
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Features ultra-avancées créées. Colonnes: {len(df.columns)}")
        return df
    
    def prepare_ultra_data(self, sequence_length: int = 20, test_size: float = 0.1, validation_size: float = 0.15, use_ultra_features: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Préparation des données ultra-avancées...")
        if use_ultra_features:
            df = self.create_ultra_features()
        else:
            df = self.data.copy()
        
        # Filtrer les colonnes non-numériques
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_columns]
        
        # Sélection intelligente des features
        base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        ultra_features = [col for col in df.columns if col not in base_features + [self.target_column]]
        
        available_base = [col for col in base_features if col in df.columns]
        available_ultra = [col for col in ultra_features if col in df.columns]
        
        # Sélection des meilleures features par corrélation
        if len(available_ultra) > 50:
            corr_matrix = df[available_ultra + [self.target_column]].corr()
            target_corr = abs(corr_matrix[self.target_column].drop(self.target_column))
            top_features = target_corr.nlargest(50).index.tolist()
            available_ultra = [f for f in available_ultra if f in top_features]
        
        all_features = available_base + available_ultra
        logger.info(f"Features sélectionnées: {len(all_features)}")
        
        dataset = df[all_features].values
        scaled_data = self.scaler.fit_transform(dataset)
        X, y = self._create_sequences_ultra(scaled_data, sequence_length)
        
        train_size = int(len(X) * (1 - test_size - validation_size))
        val_size = int(len(X) * validation_size)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"Données préparées: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        logger.info(f"Forme des données: {X_train.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _create_sequences_ultra(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, data.shape[1]-1])  # Target column
        return np.array(X), np.array(y)
    
    def build_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """Modèle Transformer pour séries temporelles"""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        attention_output = BatchNormalization()(attention_output)
        attention_output = Dropout(0.1)(attention_output)
        
        # Feed forward
        ff_output = Dense(256, activation='relu')(attention_output)
        ff_output = Dense(input_shape[1])(ff_output)
        ff_output = BatchNormalization()(ff_output)
        ff_output = Dropout(0.1)(ff_output)
        
        # Residual connection
        residual = inputs + ff_output
        
        # Global pooling
        pooled = GlobalAveragePooling1D()(residual)
        
        # Dense layers
        dense = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(pooled)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
        
        dense = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
        
        output = Dense(1)(dense)
        
        model = Model(inputs=inputs, outputs=output)
        return model
    
    def build_hybrid_model(self, input_shape: Tuple[int, int]) -> Model:
        """Modèle hybride CNN-LSTM-Transformer"""
        inputs = Input(shape=input_shape)
        
        # CNN layers
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv1D(64, 3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(2)(conv1)
        
        conv2 = Conv1D(128, 3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv1D(128, 3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(2)(conv2)
        
        # LSTM layers
        lstm1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(conv2)
        lstm1 = BatchNormalization()(lstm1)
        
        lstm2 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        
        # Attention mechanism
        attention_output, attention_weights = AttentionLayer()(lstm2)
        
        # Transformer block
        transformer_output = MultiHeadAttention(num_heads=4, key_dim=32)(attention_output, attention_output)
        transformer_output = BatchNormalization()(transformer_output)
        transformer_output = Dropout(0.1)(transformer_output)
        
        # Global pooling
        pooled = GlobalAveragePooling1D()(transformer_output)
        
        # Dense layers with regularization
        dense = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(pooled)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        
        dense = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        
        dense = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.3)(dense)
        
        output = Dense(1)(dense)
        
        model = Model(inputs=inputs, outputs=output)
        return model
    
    def build_ensemble_model(self, input_shape: Tuple[int, int]) -> Model:
        """Modèle d'ensemble avec skip connections"""
        inputs = Input(shape=input_shape)
        
        # Branch 1: LSTM with attention
        lstm_branch = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inputs)
        lstm_branch = BatchNormalization()(lstm_branch)
        lstm_branch = Dropout(0.2)(lstm_branch)
        
        attention_output, _ = AttentionLayer()(lstm_branch)
        lstm_branch = GlobalAveragePooling1D()(attention_output)
        
        # Branch 2: CNN
        conv_branch = Conv1D(128, 5, activation='relu', padding='same')(inputs)
        conv_branch = BatchNormalization()(conv_branch)
        conv_branch = MaxPooling1D(2)(conv_branch)
        conv_branch = Conv1D(64, 3, activation='relu', padding='same')(conv_branch)
        conv_branch = BatchNormalization()(conv_branch)
        conv_branch = GlobalAveragePooling1D()(conv_branch)
        
        # Branch 3: Transformer
        transformer_branch = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
        transformer_branch = BatchNormalization()(transformer_branch)
        transformer_branch = GlobalAveragePooling1D()(transformer_branch)
        
        # Concatenate branches
        concatenated = Concatenate()([lstm_branch, conv_branch, transformer_branch])
        
        # Dense layers with skip connections
        dense1 = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(concatenated)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(0.4)(dense1)
        
        # Skip connection
        skip1 = Dense(256, activation='relu')(concatenated)
        dense1 = Concatenate()([dense1, skip1])
        
        dense2 = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(0.4)(dense2)
        
        # Skip connection
        skip2 = Dense(128, activation='relu')(dense1)
        dense2 = Concatenate()([dense2, skip2])
        
        dense3 = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dense2)
        dense3 = BatchNormalization()(dense3)
        dense3 = Dropout(0.3)(dense3)
        
        output = Dense(1)(dense3)
        
        model = Model(inputs=inputs, outputs=output)
        return model
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> List[Model]:
        """Entraîne un ensemble de modèles"""
        logger.info("Entraînement de l'ensemble de modèles ultra-avancés...")
        
        models = []
        model_types = [
            ('Transformer', self.build_transformer_model),
            ('Hybrid', self.build_hybrid_model),
            ('Ensemble', self.build_ensemble_model)
        ]
        
        for i, (model_name, model_builder) in enumerate(model_types):
            logger.info(f"Construction du modèle {i+1}/3")
            logger.info(f"Construction du modèle {model_name}...")
            
            model = model_builder(X_train.shape[1:])
            
            # Compilation avec optimiseur adaptatif
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=8, min_lr=1e-6),
                ModelCheckpoint(f'best_model_{i}.h5', monitor='val_loss', save_best_only=True)
            ]
            
            # Entraînement
            logger.info(f"Entraînement du modèle {i+1}/3")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            models.append(model)
        
        logger.info("Entraînement de l'ensemble terminé")
        return models
    
    def evaluate_ensemble(self, models: List[Model], X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Évalue l'ensemble de modèles"""
        logger.info("Évaluation de l'ensemble...")
        
        # Prédictions de chaque modèle
        predictions = []
        for model in models:
            pred = model.predict(X_test, verbose=0)
            predictions.append(pred.flatten())
        
        # Prédiction d'ensemble (moyenne pondérée)
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Métriques d'évaluation
        mse = mean_squared_error(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, ensemble_pred)
        
        # MAPE
        mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        
        # Direction accuracy
        direction_accuracy = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(ensemble_pred))) * 100
        
        # Profit factor
        actual_returns = np.diff(y_test)
        predicted_returns = np.diff(ensemble_pred)
        
        correct_signals = np.sign(actual_returns) == np.sign(predicted_returns)
        profit_factor = np.sum(actual_returns[correct_signals]) / abs(np.sum(actual_returns[~correct_signals])) if np.sum(actual_returns[~correct_signals]) != 0 else 0
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy,
            'Profit_Factor': profit_factor
        }
        
        logger.info("Métriques d'évaluation de l'ensemble:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def run_ultra_analysis(self, sequence_length: int = 20) -> dict:
        """Lance l'analyse ultra-avancée complète"""
        # Récupération des données étendues
        self.data = self.fetch_extended_data(period="10y")
        
        # Préparation des données
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_ultra_data(
            sequence_length=sequence_length,
            test_size=0.1,
            validation_size=0.15,
            use_ultra_features=True
        )
        
        # Entraînement de l'ensemble
        self.models = self.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Évaluation
        metrics = self.evaluate_ensemble(self.models, X_test, y_test)
        
        return metrics
    
    def predict_next_days(self, days_ahead: int = 5) -> np.ndarray:
        """
        Prédit les prix pour les prochains jours
        
        Args:
            days_ahead (int): Nombre de jours à prédire
            
        Returns:
            np.ndarray: Prédictions des prix
        """
        if not hasattr(self, 'models') or len(self.models) == 0:
            logger.warning("Aucun modèle entraîné. Lancement de l'analyse...")
            self.run_ultra_analysis()
        
        if not hasattr(self, 'data') or self.data is None:
            logger.error("Aucune donnée disponible")
            return np.array([])
        
        try:
            # Obtenir les dernières données
            latest_data = self.data.tail(50)  # Dernières 50 observations
            
            # Créer les features
            features = self.create_ultra_features()
            features = features.tail(50)
            
            # Préparer les données pour la prédiction
            sequence_length = 20
            scaler = RobustScaler()
            
            # Sélectionner les colonnes numériques
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            features_numeric = features[numeric_columns].fillna(method='ffill').fillna(0)
            
            # Normaliser
            features_scaled = scaler.fit_transform(features_numeric)
            
            # Créer la séquence d'entrée
            input_sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
            
            # Faire les prédictions avec l'ensemble
            predictions = []
            for model in self.models:
                pred = model.predict(input_sequence, verbose=0)
                predictions.append(pred.flatten()[0])
            
            # Moyenne des prédictions
            base_prediction = np.mean(predictions)
            
            # Générer les prédictions pour les jours suivants
            future_predictions = []
            current_price = self.data['Close'].iloc[-1]
            
            for day in range(days_ahead):
                # Ajouter une variation aléatoire basée sur la volatilité historique
                volatility = self.data['Close'].pct_change().std()
                price_change = np.random.normal(0, volatility)
                
                # Ajuster selon la prédiction de base
                if day == 0:
                    predicted_price = base_prediction
                else:
                    predicted_price = future_predictions[-1] * (1 + price_change)
                
                future_predictions.append(predicted_price)
            
            return np.array(future_predictions)
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            # Retourner une prédiction simple basée sur la tendance
            current_price = self.data['Close'].iloc[-1]
            trend = self.data['Close'].pct_change().mean()
            
            predictions = []
            for day in range(days_ahead):
                predicted_price = current_price * (1 + trend * (day + 1))
                predictions.append(predicted_price)
            
            return np.array(predictions)
