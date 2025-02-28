# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# core/brokers/paper.py
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import copy
import uuid
import random

from .base import BaseBroker, Order, Position, OrderStatus
from ..utils.logger import get_logger

logger = get_logger("paper_broker")

class PaperBroker(BaseBroker):
    """Paper trading broker implementation for simulation"""
    
    def __init__(self, live_broker: BaseBroker, config: Dict[str, Any]):
        """
        Initialize paper trading broker
        
        Args:
            live_broker: Live broker for market data
            config: Dictionary containing configuration parameters
        """
        self.live_broker = live_broker  # For market data
        self.config = config
        
        # Paper trading parameters
        self.initial_capital = config.get('paper_capital', 100000.0)
        self.available_capital = self.initial_capital
        self.used_margin = 0.0
        
        # Simulation parameters
        self.slippage = config.get('paper_slippage', 0.0002)  # 0.02%
        self.commission = config.get('paper_commission', 0.0003)  # 0.03%
        
        # Trading storage
        self.positions = {}  # symbol -> Position
        self.orders = {}     # order_id -> Order
        self.order_history = []
        self.trade_history = []
        
        # Connection status
        self._connected = False
        
    async def connect(self) -> bool:
        """Connect to paper trading system"""
        try:
            # Check if live broker is connected for market data
            if not await self.live_broker.is_connected():
                logger.info("Connecting live broker for market data...")
                connected = await self.live_broker.connect()
                if not connected:
                    logger.error("Failed to connect live broker for market data")
                    return False
            
            self._connected = True
            logger.info("Paper trading system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting paper trading system: {e}")
            return False
    
    async def is_connected(self) -> bool:
        """Check if paper trading system is active"""
        # Check live broker connection for market data
        live_connected = await self.live_broker.is_connected()
        return self._connected and live_connected
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current market quote for a symbol"""
        # Use live broker for real market data
        return await self.live_broker.get_quote(symbol)
    
    async def place_order(self, 
                        symbol: str,
                        quantity: int,
                        side: str,
                        order_type: str,
                        price: float = 0.0,
                        trigger_price: float = 0.0) -> Order:
        """Place a new order in paper trading"""
        try:
            # Validate inputs
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
                
            if side not in ["BUY", "SELL"]:
                raise ValueError("Side must be BUY or SELL")
                
            if order_type not in ["MARKET", "LIMIT", "SL", "SL-M"]:
                raise ValueError("Invalid order type")
                
            if order_type in ["LIMIT", "SL"] and price <= 0:
                raise ValueError("Price must be specified for LIMIT orders")
                
            if order_type in ["SL", "SL-M"] and trigger_price <= 0:
                raise ValueError("Trigger price must be specified for SL orders")
            
            # Get current market price
            quote = await self.get_quote(symbol)
            current_price = quote['ltp']
            
            # Create order with pending status
            order_id = f"PAPER_{uuid.uuid4().hex[:8].upper()}"
            
            order = Order(
                order_id=order_id,
                symbol=symbol,
                quantity=quantity,
                price=price if order_type in ["LIMIT", "SL"] else current_price,
                order_type=order_type,
                side=side,
                status=OrderStatus.PENDING
            )
            
            self.orders[order_id] = order
            
            # Schedule order processing
            asyncio.create_task(self._process_order(order_id))
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing paper order: {e}")
            raise
    
    async def _process_order(self, order_id: str):
        """Process pending order"""
        try:
            await asyncio.sleep(0.1)  # Simulate slight delay
            
            order = self.orders[order_id]
            symbol = order.symbol
            
            # Get current price
            quote = await self.get_quote(symbol)
            current_price = quote['ltp']
            
            # Check if order can be executed
            can_execute = False
            execution_price = 0.0
            
            if order.order_type == "MARKET":
                can_execute = True
                # Apply slippage
                if order.side == "BUY":
                    execution_price = current_price * (1 + self.slippage)
                else:
                    execution_price = current_price * (1 - self.slippage)
            
            elif order.order_type == "LIMIT":
                if order.side == "BUY" and current_price <= order.price:
                    can_execute = True
                    execution_price = order.price
                elif order.side == "SELL" and current_price >= order.price:
                    can_execute = True
                    execution_price = order.price
            
            elif order.order_type == "SL":
                if order.side == "BUY" and current_price >= order.trigger_price:
                    can_execute = True
                    execution_price = max(current_price, order.price)
                elif order.side == "SELL" and current_price <= order.trigger_price:
                    can_execute = True
                    execution_price = min(current_price, order.price)
            
            elif order.order_type == "SL-M":
                if order.side == "BUY" and current_price >= order.trigger_price:
                    can_execute = True
                    # Apply slippage
                    execution_price = current_price * (1 + self.slippage)
                elif order.side == "SELL" and current_price <= order.trigger_price:
                    can_execute = True
                    # Apply slippage
                    execution_price = current_price * (1 - self.slippage)
            
            if can_execute:
                # Round price to 2 decimals
                execution_price = round(execution_price, 2)
                
                # Calculate transaction cost
                transaction_cost = execution_price * order.quantity * self.commission
                
                # Check if enough capital
                order_value = execution_price * order.quantity
                
                if order.side == "BUY" and order_value + transaction_cost > self.available_capital:
                    # Reject order due to insufficient funds
                    order.status = OrderStatus.REJECTED
                    order.execution_time = datetime.now()
                    logger.warning(f"Order {order_id} rejected: Insufficient funds")
                    return
                
                # Execute order
                order.status = OrderStatus.COMPLETE
                order.executed_price = execution_price
                order.executed_quantity = order.quantity
                order.execution_time = datetime.now()
                
                # Update capital
                if order.side == "BUY":
                    self.available_capital -= (order_value + transaction_cost)
                    self.used_margin += order_value
                else:
                    self.available_capital += (order_value - transaction_cost)
                    self.used_margin -= order_value
                
                # Update position
                await self._update_position(order)
                
                # Add to history
                self.order_history.append(copy.deepcopy(order))
                
                logger.info(f"Order {order_id} executed at {execution_price}")
            else:
                # Order remains pending
                pass
                
        except Exception as e:
            logger.error(f"Error processing order {order_id}: {e}")
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.REJECTED
    
    async def _update_position(self, order: Order):
        """Update position after order execution"""
        symbol = order.symbol
        quantity = order.quantity
        price = order.executed_price
        side = order.side
        
        # Check if position exists
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if side == "BUY":
                # Adding to position
                total_quantity = position.quantity + quantity
                position.average_price = ((position.quantity * position.average_price) + 
                                         (quantity * price)) / total_quantity
                position.quantity = total_quantity
            else:
                # Reducing position
                if quantity > position.quantity:
                    # More than we have - close position and create opposite one
                    # Calculate profit/loss
                    pnl = (price - position.average_price) * position.quantity
                    
                    # Record trade
                    self.trade_history.append({
                        'symbol': symbol,
                        'entry_price': position.average_price,
                        'exit_price': price,
                        'quantity': position.quantity,
                        'side': position.side,
                        'entry_time': position.timestamp,
                        'exit_time': order.execution_time,
                        'pnl': pnl
                    })
                    
                    # Create new opposite position with remaining quantity
                    remaining = quantity - position.quantity
                    
                    if remaining > 0:
                        # New short position
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            quantity=remaining,
                            average_price=price,
                            side="SELL"
                        )
                    else:
                        # Position closed
                        del self.positions[symbol]
                    
                else:
                    # Partial close
                    # Calculate profit/loss
                    pnl = (price - position.average_price) * quantity
                    
                    # Record trade
                    self.trade_history.append({
                        'symbol': symbol,
                        'entry_price': position.average_price,
                        'exit_price': price,
                        'quantity': quantity,
                        'side': position.side,
                        'entry_time': position.timestamp,
                        'exit_time': order.execution_time,
                        'pnl': pnl
                    })
                    
                    # Update position
                    position.quantity -= quantity
                    
                    if position.quantity == 0:
                        # Position closed
                        del self.positions[symbol]
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                side=side
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
            
        order = self.orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Order {order_id} cannot be cancelled (status: {order.status})")
            return False
            
        # Cancel order
        order.status = OrderStatus.CANCELLED
        
        # Add to history
        self.order_history.append(copy.deepcopy(order))
        
        logger.info(f"Order {order_id} cancelled")
        return True
    
    async def get_order_status(self, order_id: str) -> Order:
        """Get current status of an order"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
            
        return copy.deepcopy(self.orders[order_id])
    
    async def get_positions(self) -> List[Position]:
        """Get current open positions"""
        positions = []
        
        for symbol, position in self.positions.items():
            # Get current price for P&L calculation
            try:
                quote = await self.get_quote(symbol)
                current_price = quote['ltp']
                
                # Clone position to avoid modifying the original
                pos = copy.deepcopy(position)
                
                # Update current price and P&L
                pos.current_price = current_price
                
                if pos.side == "BUY":
                    pos.pnl = (current_price - pos.average_price) * pos.quantity
                else:
                    pos.pnl = (pos.average_price - current_price) * pos.quantity
                
                positions.append(pos)
                
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
                positions.append(copy.deepcopy(position))
        
        return positions
    
    async def get_historical_data(self,
                                symbol: str,
                                timeframe: str,
                                start_date: datetime,
                                end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical price data"""
        # Use live broker for real market data
        return await self.live_broker.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance and margin information"""
        return {
            'total_balance': self.initial_capital,
            'used_margin': self.used_margin,
            'available_margin': self.available_capital,
            'cash_balance': self.available_capital,
            'collateral_margin': 0.0
        }
    
    async def is_market_open(self) -> bool:
        """Check if market is currently open"""
        # Use live broker for real market status
        return await self.live_broker.is_market_open()
    
    async def logout(self) -> bool:
        """Logout and clean up resources"""
        self._connected = False
        return True
    
    def get_trade_history(self) -> List[Dict]:
        """Get all completed trades"""
        return self.trade_history
    
    def get_order_history(self) -> List[Order]:
        """Get all historical orders"""
        return self.order_history
    
    def reset(self):
        """Reset paper trading to initial state"""
        self.available_capital = self.initial_capital
        self.used_margin = 0.0
        self.positions = {}
        self.orders = {}
        self.order_history = []
        self.trade_history = []
        logger.info("Paper trading reset to initial state")