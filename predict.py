"""
Script para fazer previsões usando o modelo treinado
"""

import os
from datetime import datetime
import pandas as pd
from loguru import logger

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

# Importações locais
from main import CryptoDataManager, TechnicalIndicators, CryptoMLModel, Config


def predict_next_price(symbol: str = 'ETHUSDT', model_name: str = 'eth_price_predictor') -> dict:
    """
    Faz previsão do próximo preço usando dados recentes e modelo treinado

    Args:
        symbol: Par de criptomoedas
        model_name: Nome do modelo salvo

    Returns:
        Dicionário com previsão e dados atuais
    """
    config = Config()

    # Inicializar componentes
    data_manager = CryptoDataManager(config)
    ml_model = CryptoMLModel(config.model_dir)

    # Carregar modelo treinado
    if not ml_model.load_model(model_name):
        logger.error(f"Não foi possível carregar o modelo {model_name}")
        return {}

    # Buscar dados recentes (últimas 50 velas para calcular indicadores)
    logger.info(f"Buscando dados recentes para {symbol}...")
    df_recent = data_manager.fetch_klines(
        symbol=symbol,
        interval=config.default_interval,
        limit=50
    )

    if df_recent.empty:
        logger.error("Não foi possível obter dados recentes")
        return {}

    # Calcular indicadores técnicos
    df_with_indicators = TechnicalIndicators.add_all_indicators(df_recent)

    # Preparar features para previsão
    X, _ = ml_model.prepare_features(df_with_indicators, prediction_horizon=1)

    if len(X) == 0:
        logger.error("Não há dados suficientes para fazer previsão")
        return {}

    # Fazer previsão
    last_features = X.iloc[-1:].copy()
    prediction = ml_model.predict(last_features)[0]

    current_price = df_recent['Close'].iloc[-1]

    result = {
        'symbol': symbol,
        'current_price': current_price,
        'predicted_price': prediction,
        'prediction_time': datetime.now(),
        'price_change': prediction - current_price,
        'price_change_pct': ((prediction - current_price) / current_price) * 100,
        'model_features': len(ml_model.feature_columns)
    }

    return result


def analyze_prediction_accuracy(symbol: str = 'ETHUSDT', days_back: int = 7) -> dict:
    """
    Analisa a acurácia das previsões comparando com dados históricos

    Args:
        symbol: Par de criptomoedas
        days_back: Dias para análise retrospectiva

    Returns:
        Dicionário com análise de acurácia
    """
    config = Config()

    # Calcular data de início
    from datetime import timedelta
    start_date = datetime.now() - timedelta(days=days_back)

    # Inicializar componentes
    data_manager = CryptoDataManager(config)
    ml_model = CryptoMLModel(config.model_dir)

    # Carregar modelo
    if not ml_model.load_model('eth_price_predictor'):
        logger.error("Modelo não encontrado")
        return {}

    # Buscar dados históricos
    logger.info(f"Analisando acurácia para {days_back} dias...")
    df_historical = data_manager.fetch_klines(
        symbol=symbol,
        interval=config.default_interval,
        start_date=start_date
    )

    if df_historical.empty or len(df_historical) < 50:
        logger.error("Dados históricos insuficientes para análise")
        return {}

    # Adicionar indicadores
    df_with_indicators = TechnicalIndicators.add_all_indicators(df_historical)

    # Preparar dados para análise
    X, y = ml_model.prepare_features(df_with_indicators, prediction_horizon=1)

    if len(X) < 10:
        logger.error("Dados insuficientes para análise de acurácia")
        return {}

    # Fazer previsões retrospectivas
    predictions = []
    actual_values = []

    # Usar apenas os últimos 20 pontos para análise
    for i in range(min(20, len(X) - 1)):
        # Features para previsão (excluindo a última linha que seria o futuro)
        features = X.iloc[i:i+1]
        actual = y.iloc[i]

        # Fazer previsão
        pred = ml_model.predict(features)[0]

        predictions.append(pred)
        actual_values.append(actual)

    # Calcular métricas
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(actual_values, predictions)
    mse = mean_squared_error(actual_values, predictions)
    r2 = r2_score(actual_values, predictions)

    # Calcular direção correta (% de vezes que a previsão acertou a direção do movimento)
    correct_direction = 0
    for i in range(1, len(actual_values)):
        actual_direction = 1 if actual_values[i] > actual_values[i-1] else 0
        pred_direction = 1 if predictions[i] > actual_values[i-1] else 0
        if actual_direction == pred_direction:
            correct_direction += 1

    direction_accuracy = (correct_direction / (len(actual_values) - 1)) * 100

    analysis = {
        'symbol': symbol,
        'analysis_period_days': days_back,
        'predictions_made': len(predictions),
        'mae': mae,
        'mse': mse,
        'r2_score': r2,
        'direction_accuracy_pct': direction_accuracy,
        'avg_error_pct': (mae / (sum(actual_values) / len(actual_values))) * 100
    }

    return analysis


def main():
    """Função principal para demonstração"""
    logger.info("=== Sistema de Previsão de Preços de Criptomoedas ===")

    # Fazer previsão atual
    logger.info("\n1. Fazendo previsão atual...")
    prediction = predict_next_price()

    if prediction:
        logger.info("Previsão Atual:")
        logger.info(f"  Símbolo: {prediction['symbol']}")
        logger.info(f"  Preço Atual: ${prediction['current_price']:.2f}")
        logger.info(f"  Previsão: ${prediction['predicted_price']:.2f}")
        logger.info(f"  Variação: ${prediction['price_change']:+.2f} ({prediction['price_change_pct']:+.2f}%)")
        logger.info(f"  Features usadas: {prediction['model_features']}")

        # Interpretação da previsão
        if prediction['price_change_pct'] > 2:
            logger.info("  Interpretação: Sinal de alta significativo")
        elif prediction['price_change_pct'] > 0.5:
            logger.info("  Interpretação: Tendência de alta moderada")
        elif prediction['price_change_pct'] > -0.5:
            logger.info("  Interpretação: Movimento lateral")
        elif prediction['price_change_pct'] > -2:
            logger.info("  Interpretação: Tendência de baixa moderada")
        else:
            logger.info("  Interpretação: Sinal de baixa significativo")

    # Análise de acurácia
    logger.info("\n2. Analisando acurácia histórica...")
    accuracy_analysis = analyze_prediction_accuracy(days_back=3)

    if accuracy_analysis:
        logger.info("Análise de Acurácia:")
        logger.info(f"  Período: {accuracy_analysis['analysis_period_days']} dias")
        logger.info(f"  Previsões analisadas: {accuracy_analysis['predictions_made']}")
        logger.info(f"  MAE: ${accuracy_analysis['mae']:.2f}")
        logger.info(f"  R² Score: {accuracy_analysis['r2_score']:.4f}")
        logger.info(f"  Acurácia de direção: {accuracy_analysis['direction_accuracy_pct']:.1f}%")
        logger.info(f"  Erro médio percentual: {accuracy_analysis['avg_error_pct']:.2f}%")

    logger.info("\n=== Análise Concluída ===")


if __name__ == "__main__":
    main()
