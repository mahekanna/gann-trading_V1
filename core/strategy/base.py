# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# core/strategy/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging

from ..brokers.base import BaseBroker, Order, Position

class SignalType:
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"
    NO_SIGNAL = "NO_SIGNAL"

class Signal:
    def __init__(self,
                 signal_type: str,
                 symbol: str,
                 entry_price: float,
                 stop_loss: float,
                 targets: List[float],
                 quantity: int = 0):
        self.type = signal_type
        self.symbol = symbol
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.targets = targets
        self.quantity = quantity
        self.timestamp = datetime.now()

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, broker: BaseBroker, config: Dict[str, Any]):
        """
        Initialize base strategy
        
        Args:
            broker: Broker instance for order execution
            config: Strategy configuration parameters
        """
        self.broker = broker
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Strategy state
        self.is_running = False
        self.symbols = config.get('symbols', [])
        self.timeframe = config.get('timeframe', '15m')
        
        # Position tracking
        self.positions = {}  # symbol -> position info
        self.pending_orders = {}  # order_id -> order info
        
        # Performance tracking
        self.trades = []
        self.signals = []
    
    async def initialize(self) -> bool:
        """Initialize strategy"""
        try:
            if not await self.broker.is_connected():
                self.logger.error("Broker not connected")
                return False
                
            if not self.symbols:
                self.logger.error("No symbols configured")
                return False
                
            self.logger.info(f"Initializing {self.__class__.__name__} strategy")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy: {e}")
            return False
    
    async def start(self) -> bool:
        """Start strategy execution"""
        try:
            if not await self.initialize():
                return False
                
            self.is_running = True
            self.logger.info(f"Strategy {self.__class__.__name__} started")
            
            # Start monitoring
            await self._start_monitoring()
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting strategy: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop strategy execution"""
        try:
            self.is_running = False
            self.logger.info(f"Strategy {self.__class__.__name__} stopped")
            
            # Optionally close all positions
            if self.config.get('close_positions_on_stop', True):
                await self._close_all_positions()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy: {e}")
            return False
    
    async def _start_monitoring(self):
        """Start monitoring symbols"""
        self.logger.info(f"Starting to monitor {len(self.symbols)} symbols")
        
        # Start monitoring each symbol
        for symbol in self.symbols:
            asyncio.create_task(self._monitor_symbol(symbol))
    
    async def _monitor_symbol(self, symbol: str):
        """Monitor a single symbol"""
        self.logger.info(f"Monitoring {symbol}")
        
        update_interval = self.config.get('update_interval', 5)  # seconds
        
        while self.is_running:
            try:
                # Check if market is open
                is_open = await self.broker.is_market_open()
                if not is_open:
                    self.logger.debug(f"Market closed, waiting for {update_interval*10} seconds")
                    await asyncio.sleep(update_interval * 10)
                    continue
                
                # Get current quote
                quote = await self.broker.get_quote(symbol)
                
                # Check for new signals
                signal = await self.generate_signal(symbol, quote)
                
                if signal and signal.type != SignalType.NO_SIGNAL:
                    self.logger.info(f"Generated signal: {signal.type} for {symbol}")
                    # Process signal
                    await self._process_signal(signal)
                
                # Check and update positions
                if symbol in self.positions:
                    await self._update_position(symbol, quote)
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring {symbol}: {e}")
                await asyncio.sleep(update_interval)
    
    @abstractmethod
    async def generate_signal(self, symbol: str, quote: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate trading signal based on current market conditions
        
        Args:
            symbol: Trading symbol
            quote: Current market quote
            
        Returns:
            Signal instance or None if no signal
        """
        pass
    
    async def _process_signal(self, signal: Signal):
        """Process trading signal"""
        try:
            symbol = signal.symbol
            
            # Check if already have position
            has_position = symbol in self.positions
            
            if signal.type == SignalType.BUY:
                if has_position:
                    self.logger.warning(f"Already have position for {symbol}, ignoring BUY signal")
                    return
                    
                # Calculate position size if not specified
                if signal.quantity <= 0:
                    signal.quantity = self._calculate_position_size(
                        symbol, signal.entry_price, signal.stop_loss
                    )
                    
                if signal.quantity <= 0:
                    self.logger.warning(f"Zero position size for {symbol}, ignoring BUY signal")
                    return
                
                # Place buy order
                order = await self.broker.place_order(
                    symbol=symbol,
                    quantity=signal.quantity,
                    side="BUY",
                    order_type="MARKET",
                    price=0.0
                )
                
                # Track order
                self.pending_orders[order.order_id] = {
                    'order': order,
                    'signal': signal
                }
                
                # Save signal for reference
                self.signals.append(signal)
                
            elif signal.type == SignalType.SELL:
                # Similar to BUY but with SELL side
                if has_position:
                    self.logger.warning(f"Already have position for {symbol}, ignoring SELL signal")
                    return
                    
                # Calculate position size if not specified
                if signal.quantity <= 0:
                    signal.quantity = self._calculate_position_size(
                        symbol, signal.entry_price, signal.stop_loss
                    )
                    
                if signal.quantity <= 0:
                    self.logger.warning(f"Zero position size for {symbol}, ignoring SELL signal")
                    return
                
                # Place sell order
                order = await self.broker.place_order(
                    symbol=symbol,
                    quantity=signal.quantity,
                    side="SELL",
                    order_type="MARKET",
                    price=0.0
                )
                
                # Track order
                self.pending_orders[order.order_id] = {
                    'order': order,
                    'signal': signal
                }
                
                # Save signal for reference
                self.signals.append(signal)
                
            elif signal.type == SignalType.EXIT:
                if not has_position:
                    self.logger.warning(f"No position for {symbol}, ignoring EXIT signal")
                    return
                
                # Get position
                position = self.positions[symbol]
                
                # Place exit order
                exit_side = "SELL" if position['side'] == "BUY" else "BUY"
                
                order = await self.broker.place_order(
                    symbol=symbol,
                    quantity=position['quantity'],
                    side=exit_side,
                    order_type="MARKET",
                    price=0.0
                )
                
                # Track order
                self.pending_orders[order.order_id] = {
                    'order': order,
                    'signal': signal,
                    'is_exit': True
                }
                
            else:
                self.logger.warning(f"Unknown signal type: {signal.type}")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    async def _update_position(self, symbol: str, quote: Dict[str, Any]):
        """Update position with current market data"""
        try:
            if symbol not in self.positions:
                return
                
            position = self.positions[symbol]
            current_price = quote['ltp']
            
            # Update current price and P&L
            position['current_price'] = current_price
            
            if position['side'] == "BUY":
                position['pnl'] = (current_price - position['entry_price']) * position['quantity']
            else:
                position['pnl'] = (position['entry_price'] - current_price) * position['quantity']
                
            # Check stop loss
            if position['side'] == "BUY" and current_price <= position['stop_loss']:
                self.logger.info(f"Stop loss triggered for {symbol}")
                await self._exit_position(symbol, "Stop Loss")
                return
                
            if position['side'] == "SELL" and current_price >= position['stop_loss']:
                self.logger.info(f"Stop loss triggered for {symbol}")
                await self._exit_position(symbol, "Stop Loss")
                return
                
            # Check targets
            for i, target in enumerate(position['targets']):
                if i < position['targets_hit']:
                    continue  # Already hit this target
                    
                if position['side'] == "BUY" and current_price >= target:
                    self.logger.info(f"Target {i+1} hit for {symbol}")
                    await self._take_partial_profit(symbol, i, current_price)
                    return
                    
                if position['side'] == "SELL" and current_price <= target:
                    self.logger.info(f"Target {i+1} hit for {symbol}")
                    await self._take_partial_profit(symbol, i, current_price)
                    return
            
        except Exception as e:
            self.logger.error(f"Error updating position for {symbol}: {e}")
    
    async def _exit_position(self, symbol: str, reason: str):
        """Exit a position completely"""
        try:
            if symbol not in self.positions:
                self.logger.warning(f"No position for {symbol} to exit")
                return
                
            position = self.positions[symbol]
            
            # Place exit order
            exit_side = "SELL" if position['side'] == "BUY" else "BUY"
            
            order = await self.broker.place_order(
                symbol=symbol,
                quantity=position['quantity'],
                side=exit_side,
                order_type="MARKET",
                price=0.0
            )
            
            # Record trade 
            self.trades.append({
                'symbol': symbol,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'entry_price': position['entry_price'],
                'exit_price': position['current_price'],
                'quantity': position['quantity'],
                'side': position['side'],
                'pnl': position['pnl'],
                'exit_reason': reason
            })
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.info(f"Exited position for {symbol}, reason: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error exiting position for {symbol}: {e}")
    
    async def _take_partial_profit(self, symbol: str, target_index: int, current_price: float):
        """Take partial profit at target"""
        try:
            if symbol not in self.positions:
                return
                
            position = self.positions[symbol]
            
            # Calculate quantity to close
            if target_index == len(position['targets']) - 1:
                # Last target - close all remaining
                quantity = position['quantity']
            else:
                # Partial close - close a portion of position
                portion = self.config.get('target_portions', [0.33, 0.33, 0.34])
                if target_index < len(portion):
                    quantity = int(position['initial_quantity'] * portion[target_index])
                else:
                    quantity = int(position['initial_quantity'] * 0.25)  # Default 25%
                    
                quantity = min(quantity, position['quantity'])  # Don't exceed current quantity
                
            if quantity <= 0:
                self.logger.warning(f"Calculated zero quantity for partial exit, skipping")
                return
            
            # Place exit order
            exit_side = "SELL" if position['side'] == "BUY" else "BUY"
            
            order = await self.broker.place_order(
                symbol=symbol,
                quantity=quantity,
                side=exit_side,
                order_type="MARKET",
                price=0.0
            )
            
            # Record partial exit
            self.trades.append({
                'symbol': symbol,
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'quantity': quantity,
                'side': position['side'],
                'pnl': (current_price - position['entry_price']) * quantity if position['side'] == "BUY" 
                      else (position['entry_price'] - current_price) * quantity,
                'exit_reason': f"Target {target_index+1}",
                'is_partial': True
            })
            
            # Update position
            position['quantity'] -= quantity
            position['targets_hit'] = target_index + 1
            
            if position['quantity'] <= 0:
                # Position fully closed
                del self.positions[symbol]
            
            self.logger.info(f"Took partial profit at target {target_index+1} for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error taking partial profit for {symbol}: {e}")
    
    async def _check_pending_orders(self):
        """Check and update status of pending orders"""
        try:
            for order_id in list(self.pending_orders.keys()):
                order_info = self.pending_orders[order_id]
                
                # Get current order status
                updated_order = await self.broker.get_order_status(order_id)
                
                if updated_order.status == "COMPLETE":
                    # Order completed
                    symbol = updated_order.symbol
                    signal = order_info['signal']
                    
                    if order_info.get('is_exit', False):
                        # This was an exit order
                        if symbol in self.positions:
                            # Record trade
                            position = self.positions[symbol]
                            self.trades.append({
                                'symbol': symbol,
                                'entry_time': position['entry_time'],
                                'exit_time': datetime.now(),
                                'entry_price': position['entry_price'],
                                'exit_price': updated_order.executed_price,
                                'quantity': updated_order.quantity,
                                'side': position['side'],
                                'pnl': position['pnl'],
                                'exit_reason': "Order Completed"
                            })
                            
                            # Remove position
                            del self.positions[symbol]
                    else:
                        # This was an entry order
                        # Record new position
                        self.positions[symbol] = {
                            'symbol': symbol,
                            'quantity': updated_order.executed_quantity,
                            'initial_quantity': updated_order.executed_quantity,
                            'entry_price': updated_order.executed_price,
                            'current_price': updated_order.executed_price,
                            'side': updated_order.side,
                            'entry_time': datetime.now(),
                            'stop_loss': signal.stop_loss,
                            'targets': signal.targets,
                            'targets_hit': 0,
                            'pnl': 0.0
                        }
                    
                    # Remove from pending
                    del self.pending_orders[order_id]
                    
                elif updated_order.status in ["REJECTED", "CANCELLED"]:
                    # Order failed
                    self.logger.warning(f"Order {order_id} failed: {updated_order.status}")
                    del self.pending_orders[order_id]
                
        except Exception as e:
            self.logger.error(f"Error checking pending orders: {e}")
    
    async def _close_all_positions(self):
        """Close all open positions"""
        for symbol in list(self.positions.keys()):
            await self._exit_position(symbol, "Strategy Stopped")
    
    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk parameters"""
        try:
            risk_per_trade = self.config.get('risk_per_trade', 0.01)  # 1% per trade
            account_balance = self.config.get('account_balance', 100000)
            
            # Calculate risk amount
            risk_amount = account_balance * risk_per_trade
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                return 0
                
            # Calculate quantity
            quantity = int(risk_amount / risk_per_share)
            
            # Apply position limits
            max_quantity = self.config.get('max_quantity', 1000)
            quantity = min(quantity, max_quantity)
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

