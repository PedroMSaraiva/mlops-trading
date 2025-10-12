"""
Módulo de análise de dados de criptomoedas com integração de Machine Learning
"""

import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from binance.client import Client
from loguru import logger

# Importações para ML (serão usadas posteriormente)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configuração do loguru
logger.remove()  # Remove handlers padrão
logger.add(
    "crypto_ml.log",
    rotation="10 MB",
    retention="1 month",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    backtrace=True,
    diagnose=True
)
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    colorize=True
)


@dataclass
class Config:
    """Configurações da aplicação"""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    data_dir: str = "data"
    model_dir: str = "models"
    default_symbol: str = "ETHUSDT"
    default_interval: str = Client.KLINE_INTERVAL_30MINUTE
    default_limit: int = 500
    offline_mode: bool = False  # Modo offline para CI/CD


class CryptoDataManager:
    """Gerenciador de dados de criptomoedas"""

    def __init__(self, config: Config):
        self.config = config
        self.client = None
        if not config.offline_mode:
            try:
                self.client = Client(config.binance_api_key, config.binance_api_secret)
                logger.info("Cliente Binance inicializado com sucesso")
            except Exception as e:
                logger.warning(f"Não foi possível conectar à API Binance: {e}")
                logger.info("Usando modo offline - dados históricos serão carregados")
                self.config.offline_mode = True
        else:
            logger.info("Modo offline ativado - usando dados históricos")
        self._ensure_directories()

    def _ensure_directories(self):
        """Cria diretórios necessários se não existirem"""
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.model_dir, exist_ok=True)

    def fetch_klines(
        self,
        symbol: str = None,
        interval: str = None,
        limit: int = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Busca dados de klines com tratamento robusto de erros

        Args:
            symbol: Par de criptomoedas (ex: 'ETHUSDT')
            interval: Intervalo das velas
            limit: Número máximo de velas
            start_date: Data de início (opcional)
            end_date: Data de fim (opcional)

        Returns:
            DataFrame com dados das velas
        """
        symbol = symbol or self.config.default_symbol
        interval = interval or self.config.default_interval
        limit = limit or self.config.default_limit

        # Se em modo offline, tenta carregar dados salvos
        if self.config.offline_mode or self.client is None:
            logger.info("Modo offline: tentando carregar dados históricos salvos...")
            return self.load_offline_data(symbol)

        logger.info(f"Buscando {limit} velas de {interval} para {symbol}")

        try:
            if start_date and end_date:
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_date.strftime("%Y-%m-%d %H:%M:%S"),
                    end_str=end_date.strftime("%Y-%m-%d %H:%M:%S")
                )
            else:
                klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

            if not klines:
                logger.warning("Nenhum dado retornado pela API")
                return pd.DataFrame()

            df = self._process_klines_data(klines)
            logger.info(f"Dados obtidos: {len(df)} velas")
            return df

        except Exception as e:
            logger.error(f"Erro ao buscar dados: {e}")
            logger.info("Tentando carregar dados offline como fallback...")
            return self.load_offline_data(symbol)
    
    def load_offline_data(self, symbol: str = None) -> pd.DataFrame:
        """
        Carrega dados históricos salvos localmente
        
        Args:
            symbol: Par de criptomoedas (ex: 'ETHUSDT')
            
        Returns:
            DataFrame com dados históricos ou DataFrame vazio
        """
        symbol = symbol or self.config.default_symbol
        
        # Tenta carregar dados com indicadores primeiro
        indicators_file = os.path.join(self.config.data_dir, f"{symbol.lower()}_with_indicators.csv")
        raw_file = os.path.join(self.config.data_dir, f"{symbol.lower()}_raw_data.csv")
        
        # Arquivos genéricos como fallback
        generic_indicators = os.path.join(self.config.data_dir, "eth_with_indicators.csv")
        generic_raw = os.path.join(self.config.data_dir, "eth_raw_data.csv")
        
        for filepath in [indicators_file, raw_file, generic_indicators, generic_raw]:
            if os.path.exists(filepath):
                try:
                    logger.info(f"Carregando dados de: {filepath}")
                    df = pd.read_csv(filepath)
                    logger.info(f"Dados carregados com sucesso: {len(df)} linhas")
                    return df
                except Exception as e:
                    logger.warning(f"Erro ao carregar {filepath}: {e}")
        
        logger.error(f"Nenhum dado histórico encontrado para {symbol}")
        logger.info(f"Arquivos procurados: {indicators_file}, {raw_file}")
        return pd.DataFrame()

    def _process_klines_data(self, klines: list) -> pd.DataFrame:
        """Processa dados brutos de klines"""
        columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ]

        df = pd.DataFrame(klines, columns=columns)

        # Converter timestamps
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

        # Converter colunas numéricas
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
                       'Taker buy base asset volume', 'Taker buy quote asset volume']

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remover coluna desnecessária
        df = df.drop(columns=['Ignore'])

        # Adicionar coluna de símbolo para identificação
        df['Symbol'] = self.config.default_symbol

        return df

    def save_data(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> bool:
        """
        Salva dados em diferentes formatos

        Args:
            df: DataFrame a ser salvo
            filename: Nome do arquivo (sem extensão)
            format: Formato ('csv', 'json', 'parquet')

        Returns:
            True se salvou com sucesso
        """
        if df.empty:
            logger.warning("DataFrame vazio, nada para salvar")
            return False

        filepath = os.path.join(self.config.data_dir, f"{filename}.{format}")

        try:
            if format.lower() == 'csv':
                df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                df.to_json(filepath, orient='records', date_format='iso')
            elif format.lower() == 'parquet':
                df.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"Formato não suportado: {format}")

            logger.info(f"Dados salvos em: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Erro ao salvar dados: {e}")
            return False

    def load_data(self, filename: str, format: str = 'csv') -> pd.DataFrame:
        """Carrega dados salvos"""
        filepath = os.path.join(self.config.data_dir, f"{filename}.{format}")

        try:
            if format.lower() == 'csv':
                return pd.read_csv(filepath)
            elif format.lower() == 'json':
                return pd.read_json(filepath)
            elif format.lower() == 'parquet':
                return pd.read_parquet(filepath)
            else:
                raise ValueError(f"Formato não suportado: {format}")
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return pd.DataFrame()


class TechnicalIndicators:
    """Calcula indicadores técnicos para análise"""

    @staticmethod
    def add_moving_averages(df: pd.DataFrame, windows: list = [5, 10, 20, 50]) -> pd.DataFrame:
        """Adiciona médias móveis"""
        df_copy = df.copy()

        for window in windows:
            df_copy[f'SMA_{window}'] = df_copy['Close'].rolling(window=window).mean()
            df_copy[f'EMA_{window}'] = df_copy['Close'].ewm(span=window).mean()

        return df_copy

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calcula RSI (Relative Strength Index)"""
        df_copy = df.copy()

        delta = df_copy['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))

        return df_copy

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Adiciona bandas de Bollinger"""
        df_copy = df.copy()

        df_copy['BB_Middle'] = df_copy['Close'].rolling(window=window).mean()
        df_copy['BB_Std'] = df_copy['Close'].rolling(window=window).std()

        df_copy['BB_Upper'] = df_copy['BB_Middle'] + (df_copy['BB_Std'] * num_std)
        df_copy['BB_Lower'] = df_copy['BB_Middle'] - (df_copy['BB_Std'] * num_std)

        return df_copy

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona todos os indicadores técnicos"""
        df_copy = df.copy()

        # Médias móveis
        df_copy = TechnicalIndicators.add_moving_averages(df_copy)

        # RSI
        df_copy = TechnicalIndicators.add_rsi(df_copy)

        # Bandas de Bollinger
        df_copy = TechnicalIndicators.add_bollinger_bands(df_copy)

        return df_copy


class CryptoMLModel:
    """Modelo de Machine Learning para previsão de preços"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None

    def prepare_features(self, df: pd.DataFrame, target_column: str = 'Close',
                        prediction_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara features para o modelo

        Args:
            df: DataFrame com dados e indicadores técnicos
            target_column: Coluna alvo para previsão
            prediction_horizon: Horizonte de previsão (1 = próxima vela)

        Returns:
            Tuple de (X, y) onde X são features e y é target
        """
        df_copy = df.copy()

        # Criar target (preço futuro)
        df_copy['Target'] = df_copy[target_column].shift(-prediction_horizon)

        # Remover linhas com NaN (últimas linhas sem target)
        df_copy = df_copy.dropna()

        # Selecionar features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20',
            'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower'
        ]

        # Filtrar apenas colunas que existem no DataFrame
        available_features = [col for col in feature_columns if col in df_copy.columns]
        self.feature_columns = available_features

        X = df_copy[available_features]
        y = df_copy['Target']

        logger.info(f"Features preparadas: {len(X)} amostras com {len(available_features)} features")
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Treina o modelo de ML

        Args:
            X: Features
            y: Target
            test_size: Proporção do conjunto de teste

        Returns:
            Dicionário com métricas do treinamento
        """
        logger.info("Iniciando treinamento do modelo...")

        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        # Normalizar features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Treinar modelo
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Fazer previsões
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        # Calcular métricas
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        logger.info(f"Modelo treinado. R² teste: {metrics['test_r2']:.4f}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Faz previsões com o modelo treinado"""
        if self.model is None or self.scaler is None:
            raise ValueError("Modelo não foi treinado ainda")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save_model(self, filename: str) -> bool:
        """Salva o modelo treinado"""
        if self.model is None:
            logger.warning("Nenhum modelo para salvar")
            return False

        filepath = os.path.join(self.model_dir, f"{filename}.pkl")

        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }

            joblib.dump(model_data, filepath)
            logger.info(f"Modelo salvo em: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            return False

    def load_model(self, filename: str) -> bool:
        """Carrega modelo treinado"""
        filepath = os.path.join(self.model_dir, f"{filename}.pkl")

        try:
            model_data = joblib.load(filepath)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']

            logger.info(f"Modelo carregado de: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False


def main():
    """Função principal"""
    # Detectar se está rodando em ambiente CI/CD
    is_ci = os.getenv('CI') == 'true' or os.getenv('JENKINS_HOME') is not None
    
    # Configuração
    config = Config(offline_mode=is_ci)
    
    if is_ci:
        logger.info("🔧 Ambiente CI/CD detectado - usando modo offline")
    else:
        logger.info("💻 Ambiente de desenvolvimento - tentando conectar à API Binance")

    # Inicializar componentes
    data_manager = CryptoDataManager(config)
    ml_model = CryptoMLModel(config.model_dir)

    # Buscar dados
    logger.info("Buscando dados de ETHUSDT...")
    df = data_manager.fetch_klines(
        symbol='ETHUSDT',
        interval=Client.KLINE_INTERVAL_30MINUTE,
        limit=100
    )

    if df.empty:
        logger.error("Não foi possível obter dados")
        return

    # Adicionar indicadores técnicos
    logger.info("Calculando indicadores técnicos...")
    df_with_indicators = TechnicalIndicators.add_all_indicators(df)

    # Salvar dados brutos (apenas se não estiver em modo offline)
    if not config.offline_mode:
        data_manager.save_data(df, "eth_raw_data", "csv")
        data_manager.save_data(df_with_indicators, "eth_with_indicators", "csv")

    # Preparar dados para ML
    logger.info("Preparando dados para machine learning...")
    X, y = ml_model.prepare_features(df_with_indicators, prediction_horizon=1)

    if len(X) < 50:  # Precisa de dados suficientes para treinar
        logger.warning("Dados insuficientes para treinamento")
        return

    # Treinar modelo
    metrics = ml_model.train(X, y)

    # Exibir métricas
    logger.info("Métricas do modelo:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value}")

    # Salvar modelo
    ml_model.save_model("eth_price_predictor")

    # Fazer previsão para a próxima vela (demo)
    if len(X) > 0:
        last_features = X.iloc[-1:].copy()
        prediction = ml_model.predict(last_features)
        logger.info(f"Previsão para próxima vela: {prediction[0]:.2f}")
        logger.info(f"Último preço real: {y.iloc[-1]:.2f}")

    logger.info("Processo concluído!")


if __name__ == "__main__":
    main()