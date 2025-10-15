"""
API FastAPI para o sistema de previsão de preços de criptomoedas
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import time
import random
import asyncio

import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Importações locais
from main import CryptoDataManager, TechnicalIndicators, CryptoMLModel, Config

# Configuração do loguru para a API
logger.remove()
logger.add(
    "api.log",
    rotation="10 MB",
    retention="7 days",
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

# Inicializar componentes globais
config = Config()
data_manager = CryptoDataManager(config)
ml_model = CryptoMLModel(config.model_dir)

# Log de debug para verificar paths
logger.info(f"🔍 Diretório de trabalho: {os.getcwd()}")
logger.info(f"📁 Diretório de models: {config.model_dir}")
logger.info(f"📁 Caminho absoluto de models: {os.path.abspath(config.model_dir)}")
if os.path.exists(config.model_dir):
    logger.info(f"✅ Pasta models existe!")
    models_found = os.listdir(config.model_dir)
    logger.info(f"📦 Models encontrados: {models_found}")
else:
    logger.error(f"❌ Pasta models NÃO existe!")

# Modelos Pydantic para validação
class PredictionRequest(BaseModel):
    symbol: str = Field(default="ETHUSDT", description="Par de criptomoedas (ex: ETHUSDT, BTCUSDT)")
    interval: str = Field(default="30m", description="Intervalo das velas (1m, 5m, 15m, 30m, 1h, etc.)")
    limit: int = Field(default=100, ge=10, le=1000, description="Número de velas para análise")
    model_name: str = Field(default="eth_price_predictor", description="Nome do modelo treinado")

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    prediction_time: datetime
    price_change: float
    price_change_pct: float
    model_features: int
    confidence: Optional[float] = None
    interpretation: str

class ModelInfo(BaseModel):
    name: str
    features: List[str]
    training_date: Optional[datetime]
    metrics: Optional[Dict[str, float]]
    status: str

class MarketData(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: Optional[Dict[str, float]]

# Criar aplicação FastAPI
app = FastAPI(
    title="Crypto ML API",
    description="API para previsão de preços de criptomoedas usando Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Página inicial da API"""
    return """
    <html>
        <head>
            <title>Crypto ML API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                h1 { color: #333; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                code { background: #e8e8e8; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Crypto ML API</h1>
                <p>Sistema de previsão de preços de criptomoedas usando Machine Learning</p>

                <div class="endpoint">
                    <h3>📊 Previsão de Preços</h3>
                    <p><code>GET /predict</code> - Fazer previsão de preço</p>
                    <p><code>GET /predict/{symbol}</code> - Previsão para símbolo específico</p>
                </div>

                <div class="endpoint">
                    <h3>📈 Dados de Mercado</h3>
                    <p><code>GET /market/{symbol}</code> - Dados de mercado com indicadores</p>
                    <p><code>GET /indicators/{symbol}</code> - Apenas indicadores técnicos</p>
                </div>

                <div class="endpoint">
                    <h3>🤖 Modelo de ML</h3>
                    <p><code>GET /model/info</code> - Informações do modelo treinado</p>
                    <p><code>POST /model/train</code> - Treinar novo modelo</p>
                </div>

                <div class="endpoint">
                    <h3>📋 Análise</h3>
                    <p><code>GET /analysis/accuracy</code> - Análise de acurácia histórica</p>
                    <p><code>GET /analysis/backtest</code> - Backtest do modelo</p>
                </div>

                <p><a href="/docs">📖 Documentação completa da API</a></p>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Verificar saúde da API"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/predict", response_model=PredictionResponse)
async def predict_price(
    symbol: str = "ETHUSDT",
    interval: str = "30m",
    limit: int = 100,
    model_name: str = "eth_price_predictor"
):
    """
    Fazer previsão de preço para uma criptomoeda

    - **symbol**: Par de criptomoedas (ex: ETHUSDT, BTCUSDT)
    - **interval**: Intervalo das velas
    - **limit**: Número de velas para análise
    - **model_name**: Nome do modelo treinado
    """
    try:
        logger.info(f"Fazendo previsão para {symbol} com modelo {model_name}")

        # Mapear intervalos para formato da Binance
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }

        binance_interval = interval_map.get(interval.lower(), "30m")

        # Carregar modelo (remover .pkl se já estiver no nome)
        model_name_clean = model_name.replace('.pkl', '')
        if not ml_model.load_model(model_name_clean):
            raise HTTPException(status_code=404, detail=f"Modelo {model_name} não encontrado")

        # Buscar dados recentes
        df_recent = data_manager.fetch_klines(
            symbol=symbol,
            interval=getattr(data_manager.client, f"KLINE_INTERVAL_{binance_interval.upper().replace('M', 'MINUTE')}"),
            limit=limit
        )

        if df_recent.empty:
            raise HTTPException(status_code=404, detail="Não foi possível obter dados de mercado")

        # Calcular indicadores técnicos
        df_with_indicators = TechnicalIndicators.add_all_indicators(df_recent)

        # Preparar features para previsão
        X, _ = ml_model.prepare_features(df_with_indicators, prediction_horizon=1)

        if len(X) == 0:
            raise HTTPException(status_code=400, detail="Não há dados suficientes para fazer previsão")

        # Fazer previsão
        last_features = X.iloc[-1:].copy()
        prediction = ml_model.predict(last_features)[0]
        current_price = df_recent['Close'].iloc[-1]

        # Calcular métricas
        price_change = prediction - current_price
        price_change_pct = (price_change / current_price) * 100

        # Interpretação
        if price_change_pct > 2:
            interpretation = "Sinal de alta significativo"
        elif price_change_pct > 0.5:
            interpretation = "Tendência de alta moderada"
        elif price_change_pct > -0.5:
            interpretation = "Movimento lateral"
        elif price_change_pct > -2:
            interpretation = "Tendência de baixa moderada"
        else:
            interpretation = "Sinal de baixa significativo"

        result = PredictionResponse(
            symbol=symbol,
            current_price=current_price,
            predicted_price=prediction,
            prediction_time=datetime.now(),
            price_change=price_change,
            price_change_pct=price_change_pct,
            model_features=len(ml_model.feature_columns),
            interpretation=interpretation
        )

        logger.info(f"Previsão realizada: {current_price:.2f} -> {prediction:.2f} ({price_change_pct:+.2f}%)")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na previsão: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/predict/{symbol}", response_model=PredictionResponse)
async def predict_symbol(symbol: str):
    """Previsão para símbolo específico usando parâmetros padrão"""
    return await predict_price(symbol=symbol)

@app.get("/market/{symbol}", response_model=List[MarketData])
async def get_market_data(
    symbol: str = "ETHUSDT",
    interval: str = "30m",
    limit: int = 100,
    include_indicators: bool = True
):
    """Obter dados de mercado com indicadores técnicos"""
    try:
        # Mapear intervalos
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }

        binance_interval = interval_map.get(interval.lower(), "30m")

        # Buscar dados
        df = data_manager.fetch_klines(
            symbol=symbol,
            interval=getattr(data_manager.client, f"KLINE_INTERVAL_{binance_interval.upper().replace('M', 'MINUTE')}"),
            limit=limit
        )

        if df.empty:
            raise HTTPException(status_code=404, detail="Dados não encontrados")

        # Adicionar indicadores se solicitado
        if include_indicators:
            df = TechnicalIndicators.add_all_indicators(df)

        # Converter para lista de dicionários
        market_data = []
        for _, row in df.iterrows():
            indicators = {}
            if include_indicators:
                # Extrair apenas colunas de indicadores (que começam com SMA, EMA, RSI, BB)
                indicator_cols = [col for col in df.columns if any(prefix in col for prefix in ['SMA_', 'EMA_', 'RSI', 'BB_'])]
                indicators = row[indicator_cols].to_dict()

            market_data.append(MarketData(
                symbol=symbol,
                timestamp=row['Open time'],
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume'],
                indicators=indicators
            ))

        return market_data

    except Exception as e:
        logger.error(f"Erro ao buscar dados de mercado: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/indicators/{symbol}")
async def get_indicators(symbol: str = "ETHUSDT", limit: int = 100):
    """Obter apenas indicadores técnicos para um símbolo"""
    try:
        df = data_manager.fetch_klines(symbol=symbol, limit=limit)

        if df.empty:
            raise HTTPException(status_code=404, detail="Dados não encontrados")

        df_with_indicators = TechnicalIndicators.add_all_indicators(df)

        # Retornar apenas as últimas 10 velas com indicadores
        recent_data = df_with_indicators.tail(10)

        indicators_data = []
        for _, row in recent_data.iterrows():
            # Extrair indicadores
            indicators = {}
            indicator_cols = [col for col in df_with_indicators.columns
                            if any(prefix in col for prefix in ['SMA_', 'EMA_', 'RSI', 'BB_'])]

            for col in indicator_cols:
                if pd.notna(row[col]):
                    indicators[col] = float(row[col])

            indicators_data.append({
                "timestamp": row['Open time'].isoformat(),
                "close_price": float(row['Close']),
                "indicators": indicators
            })

        return {"symbol": symbol, "indicators": indicators_data}

    except Exception as e:
        logger.error(f"Erro ao buscar indicadores: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Obter informações sobre o modelo treinado"""
    try:
        # Tentar carregar o modelo padrão
        model_loaded = ml_model.load_model("eth_price_predictor")

        if not model_loaded:
            return ModelInfo(
                name="eth_price_predictor",
                features=[],
                training_date=None,
                metrics=None,
                status="not_trained"
            )

        # Obter informações do modelo
        features = ml_model.feature_columns or []
        training_date = None

        # Tentar obter data de criação do arquivo
        model_path = os.path.join(config.model_dir, "eth_price_predictor.pkl")
        if os.path.exists(model_path):
            training_date = datetime.fromtimestamp(os.path.getctime(model_path))

        return ModelInfo(
            name="eth_price_predictor",
            features=features,
            training_date=training_date,
            metrics=None,  # Podemos adicionar métricas se salvarmos junto com o modelo
            status="loaded"
        )

    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/model/train")
async def train_model(
    background_tasks: BackgroundTasks,
    symbol: str = "ETHUSDT",
    interval: str = "30m",
    limit: int = 500
):
    """Treinar um novo modelo (execução em background)"""
    try:
        background_tasks.add_task(_train_model_background, symbol, interval, limit)

        return {
            "message": "Treinamento iniciado em background",
            "symbol": symbol,
            "status": "training_started"
        }

    except Exception as e:
        logger.error(f"Erro ao iniciar treinamento: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

async def _train_model_background(symbol: str, interval: str, limit: int):
    """Função em background para treinar o modelo"""
    try:
        logger.info(f"Iniciando treinamento em background para {symbol}")

        # Mapear intervalos
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d"
        }

        binance_interval = interval_map.get(interval.lower(), "30m")

        # Buscar dados
        df = data_manager.fetch_klines(
            symbol=symbol,
            interval=getattr(data_manager.client, f"KLINE_INTERVAL_{binance_interval.upper().replace('M', 'MINUTE')}"),
            limit=limit
        )

        if df.empty:
            logger.error("Não foi possível obter dados para treinamento")
            return

        # Adicionar indicadores
        df_with_indicators = TechnicalIndicators.add_all_indicators(df)

        # Preparar dados para ML
        X, y = ml_model.prepare_features(df_with_indicators, prediction_horizon=1)

        if len(X) < 50:
            logger.warning("Dados insuficientes para treinamento")
            return

        # Treinar modelo
        metrics = ml_model.train(X, y)

        # Salvar modelo
        ml_model.save_model(f"{symbol.lower()}_price_predictor")

        logger.info(f"Treinamento concluído para {symbol}. R²: {metrics.get('test_r2', 'N/A')}")

    except Exception as e:
        logger.error(f"Erro no treinamento em background: {e}")

@app.get("/analysis/accuracy")
async def get_accuracy_analysis(
    symbol: str = "ETHUSDT",
    days_back: int = 7
):
    """Análise de acurácia histórica do modelo"""
    try:
        # Importar função do predict.py
        from predict import analyze_prediction_accuracy

        analysis = analyze_prediction_accuracy(symbol, days_back)

        if not analysis:
            raise HTTPException(status_code=404, detail="Não foi possível realizar análise")

        return analysis

    except ImportError:
        raise HTTPException(status_code=500, detail="Erro interno na análise")
    except Exception as e:
        logger.error(f"Erro na análise de acurácia: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


# ==================== SISTEMA DE LOGS PARA LOKI ====================

class LokiLogger:
    """Classe para enviar logs estruturados ao Loki"""
    
    LOKI_URL = "http://loki-stack.monitoring.svc.cluster.local:3100/loki/api/v1/push"
    
    @staticmethod
    def send_log(level: str, message: str, extra_data: Optional[Dict] = None) -> bool:
        """
        Envia um log ao Loki
        
        Args:
            level: Nível do log (INFO, WARNING, ERROR)
            message: Mensagem do log
            extra_data: Dados adicionais em formato dict
            
        Returns:
            bool: True se enviado com sucesso, False caso contrário
        """
        try:
            timestamp_ns = str(int(time.time() * 1e9))
            
            # Preparar dados do log
            log_data = {
                "level": level,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Adicionar dados extras se fornecidos
            if extra_data:
                log_data.update(extra_data)
            
            payload = {
                "streams": [
                    {
                        "stream": {
                            "job": "crypto_ml_api",
                            "level": level.lower()
                        },
                        "values": [
                            [timestamp_ns, json.dumps(log_data)]
                        ]
                    }
                ]
            }
            
            response = requests.post(
                LokiLogger.LOKI_URL,
                json=payload,
                timeout=5
            )
            
            return response.status_code == 204
            
        except Exception as e:
            logger.error(f"Erro ao enviar log para Loki: {e}")
            return False


# Variável global para controlar o background task
_metrics_task_running = False


async def periodic_metrics_logger():
    """
    Task em background que envia logs aleatórios para o Loki a cada 1 minuto
    Simula diferentes cenários: INFO, WARNING, ERROR e predições
    """
    global _metrics_task_running
    
    if _metrics_task_running:
        logger.warning("Task de métricas já está em execução")
        return
    
    _metrics_task_running = True
    logger.info("🚀 Iniciando envio periódico de logs para Loki (1 minuto)")
    
    log_types = ["info", "warning", "error", "predict"]
    symbols = ["ETHUSDT", "BTCUSDT", "BNBUSDT", "ADAUSDT"]
    
    try:
        while _metrics_task_running:
            # Escolher tipo de log aleatório
            log_type = random.choice(log_types)
            symbol = random.choice(symbols)
            
            if log_type == "info":
                # Log de informação
                messages = [
                    f"Sistema de predição operacional para {symbol}",
                    f"Análise de mercado concluída para {symbol}",
                    f"Indicadores técnicos calculados para {symbol}",
                    f"Conexão com Binance API estável - {symbol}",
                    f"Cache de dados atualizado para {symbol}"
                ]
                message = random.choice(messages)
                
                LokiLogger.send_log(
                    level="INFO",
                    message=message,
                    extra_data={
                        "symbol": symbol,
                        "operation": "system_check",
                        "status": "ok"
                    }
                )
                logger.info(f"📊 [LOKI] {message}")
                
            elif log_type == "warning":
                # Log de aviso
                messages = [
                    f"Alta volatilidade detectada em {symbol}",
                    f"Volume de negociação abaixo da média para {symbol}",
                    f"Indicador RSI próximo de sobrecompra em {symbol}",
                    f"Latência elevada na API Binance para {symbol}",
                    f"Divergência entre indicadores detectada em {symbol}"
                ]
                message = random.choice(messages)
                
                LokiLogger.send_log(
                    level="WARNING",
                    message=message,
                    extra_data={
                        "symbol": symbol,
                        "operation": "market_analysis",
                        "severity": "medium"
                    }
                )
                logger.warning(f"⚠️  [LOKI] {message}")
                
            elif log_type == "error":
                # Log de erro
                messages = [
                    f"Falha ao conectar com Binance API para {symbol}",
                    f"Timeout ao buscar dados de mercado de {symbol}",
                    f"Erro ao calcular indicadores técnicos para {symbol}",
                    f"Modelo de ML não encontrado para {symbol}",
                    f"Dados insuficientes para predição de {symbol}"
                ]
                message = random.choice(messages)
                
                LokiLogger.send_log(
                    level="ERROR",
                    message=message,
                    extra_data={
                        "symbol": symbol,
                        "operation": "data_fetch",
                        "error_type": "api_error",
                        "severity": "high"
                    }
                )
                logger.error(f"❌ [LOKI] {message}")
                
            else:  # predict
                # Log de predição
                current_price = round(random.uniform(2000, 3000), 2)
                predicted_price = round(current_price * random.uniform(0.98, 1.02), 2)
                change_pct = round(((predicted_price - current_price) / current_price) * 100, 2)
                confidence = round(random.uniform(0.7, 0.95), 2)
                
                message = f"Predição realizada para {symbol}: ${current_price:.2f} → ${predicted_price:.2f} ({change_pct:+.2f}%)"
                
                LokiLogger.send_log(
                    level="INFO",
                    message=message,
                    extra_data={
                        "symbol": symbol,
                        "operation": "predict",
                        "current_price": current_price,
                        "predicted_price": predicted_price,
                        "price_change_pct": change_pct,
                        "confidence": confidence,
                        "model": "ml_predictor_v1"
                    }
                )
                logger.info(f"🤖 [LOKI] {message}")
            
            # Aguardar 1 minuto antes do próximo envio
            await asyncio.sleep(60)
            
    except asyncio.CancelledError:
        logger.info("Task de métricas cancelada")
    except Exception as e:
        logger.error(f"Erro na task periódica de logs: {e}")
    finally:
        _metrics_task_running = False


@app.on_event("startup")
async def startup_event():
    """Evento executado ao iniciar a aplicação"""
    logger.info("🎯 API inicializada - Configurando tarefas em background")
    # Iniciar task de envio periódico de logs
    asyncio.create_task(periodic_metrics_logger())


@app.on_event("shutdown")
async def shutdown_event():
    """Evento executado ao desligar a aplicação"""
    global _metrics_task_running
    _metrics_task_running = False
    logger.info("🛑 Aplicação encerrada - Tasks finalizadas")


@app.get("/metrics")
async def get_metrics_status():
    """
    Endpoint para verificar status do sistema de métricas
    
    Returns:
        Status do envio de logs para Loki
    """
    return {
        "status": "active" if _metrics_task_running else "inactive",
        "loki_url": LokiLogger.LOKI_URL,
        "interval": "60 seconds",
        "log_types": ["INFO", "WARNING", "ERROR", "PREDICT"],
        "description": "Logs são enviados automaticamente para o Loki a cada 1 minuto"
    }


@app.post("/metrics/manual")
async def send_manual_log(
    level: str = Query("INFO", description="Nível do log: INFO, WARNING ou ERROR"),
    message: str = Query(..., description="Mensagem do log"),
    symbol: str = Query("ETHUSDT", description="Símbolo da criptomoeda")
):
    """
    Endpoint para envio manual de log ao Loki
    
    Args:
        level: Nível do log (INFO, WARNING, ERROR)
        message: Mensagem do log
        symbol: Símbolo da crypto
    """
    if level.upper() not in ["INFO", "WARNING", "ERROR"]:
        raise HTTPException(status_code=400, detail="Level deve ser INFO, WARNING ou ERROR")
    
    success = LokiLogger.send_log(
        level=level.upper(),
        message=message,
        extra_data={"symbol": symbol, "source": "manual"}
    )
    
    if success:
        logger.info(f"Log manual enviado ao Loki: [{level}] {message}")
        return {"status": "success", "message": "Log enviado ao Loki com sucesso"}
    else:
        raise HTTPException(status_code=500, detail="Erro ao enviar log para Loki")


# ==================== FIM DO SISTEMA DE LOGS ====================


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Iniciando servidor FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
