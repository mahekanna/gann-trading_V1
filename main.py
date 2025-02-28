# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# main.py
#!/usr/bin/env python3
import asyncio
import argparse
import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("main")

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Store for broker and engine references
broker = None
trading_engine = None

async def start_trading(config_path: str, mode: str) -> bool:
    """Start the trading system"""
    global broker, trading_engine
    
    try:
        logger.info(f"Starting trading system in {mode} mode...")
        
        # Load configuration
        config = load_config(config_path)
        if not config:
            return False
        
        # Initialize broker
        if mode == "paper":
            # For paper trading, we still need a real broker for market data
            from core.brokers.icici import ICICIDirectBroker
            
            live_broker = ICICIDirectBroker(config)
            connected = await live_broker.connect()
            
            if not connected:
                logger.error("Failed to connect to broker for market data")
                return False
                
            # Create paper broker
            from core.brokers.paper import PaperBroker
            broker = PaperBroker(live_broker, config)
            
            logger.info("Paper trading broker initialized")
            
        elif mode == "live":
            # For live trading, use the real broker directly
            from core.brokers.icici import ICICIDirectBroker
            
            broker = ICICIDirectBroker(config)
            connected = await broker.connect()
            
            if not connected:
                logger.error("Failed to connect to broker")
                return False
                
            logger.info("Live trading broker initialized")
            
        else:
            logger.error(f"Unknown mode: {mode}")
            return False
        
        # Initialize trading engine
        from core.engine.execution import TradingEngine
        
        trading_engine = TradingEngine(broker, config)
        
        # Create and add strategies
        from core.strategy.gann import GannSquareStrategy
        
        # Create strategy for each configured symbol set
        for strategy_config in config.get('strategies', []):
            strategy = GannSquareStrategy(broker, strategy_config)
            trading_engine.add_strategy(strategy)
        
        # Initialize and start engine
        initialized = await trading_engine.initialize()
        if not initialized:
            logger.error("Failed to initialize trading engine")
            return False
            
        started = await trading_engine.start()
        if not started:
            logger.error("Failed to start trading engine")
            return False
            
        logger.info("Trading system started successfully")
        
        # Wait for termination
        termination_event = asyncio.Event()
        
        try:
            await termination_event.wait()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping...")
        finally:
            await stop_trading()
        
        return True
        
    except Exception as e:
        logger.error(f"Error starting trading system: {e}")
        return False

async def stop_trading() -> None:
    """Stop the trading system"""
    global trading_engine
    
    if trading_engine:
        await trading_engine.stop()
        logger.info("Trading system stopped")

def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate required fields
        required_fields = ['api_key', 'api_secret']
        
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required config field: {field}")
                return None
                
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

async def run_backtest(config_path: str, symbol: str, start_date: str, end_date: str) -> bool:
    """Run a backtest"""
    try:
        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # Load configuration
        config = load_config(config_path)
        if not config:
            return False
        
        # Convert dates
        from datetime import datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Create backtest engine
        from core.engine.backtest import BacktestEngine
        
        backtest = BacktestEngine(config)
        
        # Load data
        await backtest.load_data([symbol], start, end, "15m")
        
        # Create strategy config
        strategy_config = config.get('strategies', [{}])[0].copy()
        strategy_config['symbols'] = [symbol]
        
        # Add strategy
        from core.strategy.gann import GannSquareStrategy
        backtest.add_strategy(GannSquareStrategy, strategy_config)
        
        # Run backtest
        results = await backtest.run()
        
        # Output results
        print("\n=== Backtest Results ===")
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ₹{results['initial_capital']:,.2f}")
        print(f"Final Equity: ₹{results['final_equity']:,.2f}")
        print(f"Return: {results['return_pct']:,.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:,.2f}%")
        print(f"Win Rate: {results['win_rate']:,.2f}%")
        print(f"Profit Factor: {results['profit_factor']:,.2f}")
        print(f"Total Trades: {results['total_trades']}")
        
        # Save results to file
        results_path = f"backtest_results_{symbol}_{start_date}_to_{end_date}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
            
        logger.info(f"Backtest results saved to {results_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return False

async def run_terminal_ui() -> None:
    """Run terminal UI"""
    try:
        # Import terminal UI
        from ui.terminal.app import GannTradingApp
        
        # Create and run app
        app = GannTradingApp()
        await app.run_async()
        
    except Exception as e:
        logger.error(f"Error running terminal UI: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Gann Trading System")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Trading command
    trade_parser = subparsers.add_parser("trade", help="Start trading")
    trade_parser.add_argument("--config", default="config/trading_config.json", help="Path to config file")
    trade_parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Trading mode")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--config", default="config/trading_config.json", help="Path to config file")
    backtest_parser.add_argument("--symbol", required=True, help="Symbol to backtest")
    backtest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Run terminal UI")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate command
    if args.command == "trade":
        asyncio.run(start_trading(args.config, args.mode))
    elif args.command == "backtest":
        asyncio.run(run_backtest(args.config, args.symbol, args.start, args.end))
    elif args.command == "ui":
        asyncio.run(run_terminal_ui())
    else:
        # Default to UI
        asyncio.run(run_terminal_ui())

if __name__ == "__main__":
    main()