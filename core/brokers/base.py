# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# core/brokers/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

class OrderStatus:
    PENDING = "PENDING"
    COMPLETE = "COMPLETE" 
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class Order:
    def __init__(self, 
                 order_id: str,
                 symbol: str,
                 quantity: int,
                 price: float,
                 order_type: str,
                 side: str,
                 status: str = OrderStatus.PENDING):
        self.order_id = order_id
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.order_type = order_type
        self.side = side
        self.status = status
        self.executed_price = 0.0
        self.executed_quantity = 0
        self.timestamp = datetime.now()
        self.execution_time = None

class Position:
    def __init__(self,
                 symbol: str,
                 quantity: int,
                 average_price: float,
                 side: str):
        self.symbol = symbol
        self.quantity = quantity
        self.average_price = average_price
        self.side = side
        self.current_price = average_price
        self.pnl = 0.0
        self.timestamp = datetime.now()

class BaseBroker(ABC):
    """Abstract base class for all broker implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection with broker"""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if broker connection is active"""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current market quote for a symbol"""
        pass
    
    @abstractmethod
    async def place_order(self, 
                        symbol: str,
                        quantity: int,
                        side: str,
                        order_type: str,
                        price: float = 0.0,
                        trigger_price: float = 0.0) -> Order:
        """Place a new order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get current status of an order"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current open positions"""
        pass
    
    @abstractmethod
    async def get_historical_data(self,
                                symbol: str,
                                timeframe: str,
                                start_date: datetime,
                                end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical price data"""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance and margin information"""
        pass
    
    @abstractmethod
    async def is_market_open(self) -> bool:
        """Check if market is currently open"""
        pass
    
    @abstractmethod
    async def logout(self) -> bool:
        """Logout and clean up resources"""
        pass


