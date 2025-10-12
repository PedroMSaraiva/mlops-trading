# 🚀 Crypto ML - Sistema de Previsão de Criptomoedas

Sistema completo para análise de dados de criptomoedas e previsão de preços usando Machine Learning.

## 📋 Pré-requisitos

- Python 3.13+
- uv (recomendado) ou pip

## 🔄 Modo Offline (CI/CD)

O sistema detecta automaticamente quando está rodando em ambiente CI/CD (Jenkins, GitHub Actions, etc.) e **utiliza modo offline**, carregando dados históricos salvos na pasta `data/` ao invés de conectar à API da Binance.

**Como funciona:**
- ✅ Detecta variáveis de ambiente `CI=true` ou `JENKINS_HOME`
- ✅ Carrega dados de `data/eth_with_indicators.csv` ou `data/eth_raw_data.csv`
- ✅ Treina o modelo com dados históricos
- ✅ Não requer acesso à API da Binance durante o pipeline

**Para forçar modo offline manualmente:**
```bash
export CI=true
python main.py
```

## ⚡ Iniciação Rápida

### 1. Instalar dependências
```bash
# Clone o projeto
git clone <url-do-repositorio>
cd mlops

# Instalar dependências com uv (recomendado)
uv sync
```

### 2. Executar análise inicial
```bash
# Treinar modelo inicial (conecta à API Binance em dev)
uv run main.py

# Ou em modo offline
CI=true uv run main.py
```

### 3. Subir API FastAPI
```bash
# Iniciar servidor API
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Acessar funcionalidades

**Interface web da API:**
- 📖 Documentação: http://localhost:8000/docs
- 🏠 Página inicial: http://localhost:8000

**Exemplos de uso via API:**
```bash
# Verificar se API está rodando
curl http://localhost:8000/health

# Previsão de preço do Ethereum
curl "http://localhost:8000/predict?symbol=ETHUSDT"

# Dados de mercado do Bitcoin
curl "http://localhost:8000/market/BTCUSDT?limit=50"
```

**Script de previsão local:**
```bash
# Análise detalhada e previsões
uv run predict.py
```

## 🛠️ Estrutura do Projeto

```
mlops/
├── main.py       # Análise inicial e treinamento
├── predict.py    # Previsões detalhadas
├── api.py        # API FastAPI
├── data/         # Dados salvos
├── models/       # Modelos treinados
└── logs/         # Arquivos de log
```

## 🔧 Funcionalidades

### 📊 Análise de Criptomoedas
- ✅ Dados históricos da Binance
- ✅ Indicadores técnicos (RSI, médias móveis, Bollinger)
- ✅ Modelo Random Forest para previsões
- ✅ Métricas de performance detalhadas

### 🌐 API FastAPI
- ✅ Interface web interativa (`/docs`)
- ✅ Endpoints para previsões e dados
- ✅ Validação automática de parâmetros
- ✅ Logs estruturados com Loguru

## 📈 Exemplo de Uso

```bash
# 1. Treinar modelo
uv run main.py

# 2. Ver previsões detalhadas
uv run predict.py

# 3. Usar API web
uv run uvicorn api:app --reload
# Acesse: http://localhost:8000
```

## 🚨 Notas Importantes

- **Dados públicos**: Funciona sem chave da Binance
- **Para produção**: Configure variáveis de ambiente para dados privados
- **Logs**: Verifique `crypto_ml.log` e `api.log` para debug
- **Disclaimer**: Uso educacional apenas, não para investimentos

---

**🚀 Projeto desenvolvido para fins educacionais**
