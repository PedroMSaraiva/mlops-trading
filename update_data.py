#!/usr/bin/env python3
"""
Script para atualizar dados históricos da Binance
Execute localmente para atualizar os dados que serão usados no CI/CD
"""

import os
from main import Config, CryptoDataManager, TechnicalIndicators
from binance.client import Client
from loguru import logger

def update_historical_data():
    """Atualiza dados históricos locais"""
    
    # Força modo online
    config = Config(offline_mode=False)
    
    try:
        data_manager = CryptoDataManager(config)
        
        if data_manager.client is None:
            logger.error("❌ Não foi possível conectar à API Binance")
            logger.info("Verifique se você está em uma região permitida pela Binance")
            logger.info("Ou configure suas credenciais API se necessário")
            return False
        
        logger.info("🔄 Buscando dados atualizados de ETHUSDT...")
        
        # Buscar mais dados históricos para melhor treinamento
        df = data_manager.fetch_klines(
            symbol='ETHUSDT',
            interval=Client.KLINE_INTERVAL_30MINUTE,
            limit=1000  # Mais dados para melhor modelo
        )
        
        if df.empty:
            logger.error("❌ Não foi possível obter dados")
            return False
        
        logger.info(f"✅ {len(df)} velas obtidas")
        
        # Adicionar indicadores técnicos
        logger.info("📊 Calculando indicadores técnicos...")
        df_with_indicators = TechnicalIndicators.add_all_indicators(df)
        
        # Salvar dados
        logger.info("💾 Salvando dados...")
        data_manager.save_data(df, "eth_raw_data", "csv")
        data_manager.save_data(df_with_indicators, "eth_with_indicators", "csv")
        
        logger.info("✅ Dados atualizados com sucesso!")
        logger.info(f"📁 Arquivos salvos em: {config.data_dir}/")
        logger.info("💡 Faça commit desses arquivos para usar no CI/CD")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar dados: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🔄 Atualizador de Dados Históricos")
    logger.info("=" * 60)
    logger.info("")
    
    success = update_historical_data()
    
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✅ Processo concluído com sucesso!")
        logger.info("")
        logger.info("📝 Próximos passos:")
        logger.info("   1. Revise os arquivos em data/")
        logger.info("   2. git add data/*.csv")
        logger.info("   3. git commit -m 'Update historical data'")
        logger.info("   4. git push")
    else:
        logger.info("❌ Falha ao atualizar dados")
        logger.info("")
        logger.info("💡 Dicas:")
        logger.info("   - Verifique sua conexão com internet")
        logger.info("   - Confirme que você está em uma região permitida pela Binance")
        logger.info("   - Use VPN se necessário")
    logger.info("=" * 60)
