# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# core/engine/execution.py
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from ..brokers.base import BaseBroker
from ..strategy.base import BaseStrategy
from ..utils.logger import get_logger

logger = get_logger("trading_engine")

class EngineState(Enum):
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

class TradingEngine:
    """Main trading engine that coordinates broker and strategies"""
    
    def __init__(self, broker: BaseBroker, config: Dict[str, Any]):
        """
        Initialize trading engine
        
        Args:
            broker: Broker instance for order execution
            config: Trading engine configuration
        """
        self.broker = broker
        self.config = config
        self.strategies = []
        self.state = EngineState.INITIALIZING
        
        # Engine settings
        self.check_interval = config.get('check_interval', 1)  # seconds
        self.market_check_interval = config.get('market_check_interval', 60)  # seconds
        
        # Trading hours
        self.trading_start = config.get('trading_start', '09:15')
        self.trading_end = config.get('trading_end', '15:30')
        self.square_off_time = config.get('square_off_time', '15:15')
        
        # Tasks
        self.main_task = None
        self.market_check_task = None
        
        # Performance monitoring
        self.cycle_count = 0
        self.error_count = 0
        self.last_status_log = datetime.now()
        self.status_log_interval = config.get('status_log_interval', 300)  # seconds
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add a trading strategy to the engine"""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.__class__.__name__}")
    
    async def initialize(self) -> bool:
        """Initialize trading engine and all components"""
        try:
            logger.info("Initializing trading engine...")
            
            # Connect to broker
            if not await self.broker.is_connected():
                connected = await self.broker.connect()
                if not connected:
                    logger.error("Failed to connect to broker")
                    self.state = EngineState.ERROR
                    return False
            
            # Initialize strategies
            for strategy in self.strategies:
                initialized = await strategy.initialize()
                if not initialized:
                    logger.error(f"Failed to initialize strategy: {strategy.__class__.__name__}")
                    self.state = EngineState.ERROR
                    return False
            
            self.state = EngineState.READY
            logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading engine: {e}")
            self.state = EngineState.ERROR
            return False
    
    async def start(self) -> bool:
        """Start trading engine"""
        try:
            if self.state != EngineState.READY:
                logger.error(f"Cannot start engine in {self.state} state")
                return False
                
            logger.info("Starting trading engine...")
            self.state = EngineState.RUNNING
            
            # Start main loop task
            self.main_task = asyncio.create_task(self._main_loop())
            
            # Start market check task
            self.market_check_task = asyncio.create_task(self._check_market_status())
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            self.state = EngineState.ERROR
            return False
    
    async def stop(self) -> bool:
        """Stop trading engine"""
        try:
            if self.state not in [EngineState.RUNNING, EngineState.ERROR]:
                logger.error(f"Cannot stop engine in {self.state} state")
                return False
                
            logger.info("Stopping trading engine...")
            self.state = EngineState.STOPPING
            
            # Stop strategies
            for strategy in self.strategies:
                if strategy.is_running:
                    await strategy.stop()
                    
            # Cancel tasks
            if self.main_task:
                self.main_task.cancel()
                try:
                    await self.main_task
                except asyncio.CancelledError:
                    pass
                    
            if self.market_check_task:
                self.market_check_task.cancel()
                try:
                    await self.market_check_task
                except asyncio.CancelledError:
                    pass
            
            self.state = EngineState.STOPPED
            logger.info("Trading engine stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
            self.state = EngineState.ERROR
            return False
    
    async def _main_loop(self):
        """Main trading engine loop"""
        try:
            while self.state == EngineState.RUNNING:
                try:
                    loop_start = datetime.now()
                    
                    # Start or check strategies
                    await self._process_strategies()
                    
                    # Log periodic status
                    await self._log_status_if_needed()
                    
                    # Update cycle count
                    self.cycle_count += 1
                    
                    # Calculate sleep time (target fixed cycle time)
                    loop_duration = (datetime.now() - loop_start).total_seconds()
                    sleep_time = max(0, self.check_interval - loop_duration)
                    
                    await asyncio.sleep(sleep_time)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    self.error_count += 1
                    await asyncio.sleep(self.check_interval)
                    
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            self.state = EngineState.ERROR
            
        logger.info("Main loop exited")
    
    async def _check_market_status(self):
        """Check market status periodically"""
        try:
            while self.state == EngineState.RUNNING:
                try:
                    # Check if market is open
                    is_open = await self.broker.is_market_open()
                    
                    # Check square off time
                    current_time = datetime.now().time()
                    square_off_time = datetime.strptime(self.square_off_time, "%H:%M").time()
                    
                    if current_time >= square_off_time and self._has_running_strategies():
                        logger.info("Square off time reached, stopping strategies")
                        for strategy in self.strategies:
                            if strategy.is_running:
                                await strategy.stop()
                    
                    # Check if we need to start trading
                    if is_open and current_time < square_off_time and not self._has_running_strategies():
                        logger.info("Market is open, starting strategies")
                        for strategy in self.strategies:
                            if not strategy.is_running:
                                await strategy.start()
                    
                    # Sleep for next check
                    await asyncio.sleep(self.market_check_interval)
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error checking market status: {e}")
                    await asyncio.sleep(self.market_check_interval)
                    
        except asyncio.CancelledError:
            logger.info("Market check task cancelled")
        except Exception as e:
            logger.error(f"Fatal error in market check: {e}")
    
    def _has_running_strategies(self) -> bool:
        """Check if any strategies are running"""
        return any(strategy.is_running for strategy in self.strategies)
    
    async def _process_strategies(self):
        """Process all strategies"""
        for strategy in self.strategies:
            if not strategy.is_running:
                continue
                
            try:
                # Nothing to do here - strategies run their own monitoring
                pass
            except Exception as e:
                logger.error(f"Error processing strategy {strategy.__class__.__name__}: {e}")
    
    async def _log_status_if_needed(self):
        """Log status information periodically"""
        now = datetime.now()
        if (now - self.last_status_log).total_seconds() >= self.status_log_interval:
            # Log status
            logger.info("=== Trading Engine Status ===")
            logger.info(f"State: {self.state.value}")
            logger.info(f"Cycles: {self.cycle_count}")
            logger.info(f"Errors: {self.error_count}")
            logger.info(f"Strategies: {len(self.strategies)} (Running: {sum(1 for s in self.strategies if s.is_running)})")
            
            # Log account info
            try:
                account = await self.broker.get_account_balance()
                logger.info(f"Account Balance: {account.get('total_balance', 0)}")
                logger.info(f"Used Margin: {account.get('used_margin', 0)}")
                logger.info(f"Available Margin: {account.get('available_margin', 0)}")
            except Exception as e:
                logger.error(f"Error getting account info: {e}")
            
            logger.info("=============================")
            
            self.last_status_log = now
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'state': self.state.value,
            'cycles': self.cycle_count,
            'errors': self.error_count,
            'strategies': [
                {
                    'name': strategy.__class__.__name__,
                    'running': strategy.is_running,
                    'symbols': strategy.symbols,
                    'positions': len(strategy.positions)
                }
                for strategy in self.strategies
            ],
            'timestamp': datetime.now()
        }

