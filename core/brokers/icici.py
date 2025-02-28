# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# core/brokers/icici.py
import os
import json
import pyotp
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
from breeze_connect import BreezeConnect

from .base import BaseBroker, Order, Position, OrderStatus
from ..utils.logger import get_logger

logger = get_logger("icici_broker")

class SessionManager:
    """Manages ICICI Breeze API session"""
    
    def __init__(self, 
                api_key: str,
                api_secret: str,
                totp_secret: Optional[str] = None,
                session_file: str = "session.json"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.totp_secret = totp_secret
        self.session_file = session_file
        self.breeze = None
        self.session_token = None
        self.last_login = None
    
    async def load_session(self) -> Optional[str]:
        """Load session token from file if valid"""
        try:
            if not os.path.exists(self.session_file):
                logger.info("No saved session found")
                return None
                
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
                
            token = session_data.get('token')
            timestamp = session_data.get('timestamp')
            
            if not token or not timestamp:
                return None
                
            # Check if session is still valid (less than 24 hours old)
            login_time = datetime.fromisoformat(timestamp)
            if datetime.now() - login_time > timedelta(hours=24):
                logger.info("Saved session expired")
                return None
                
            return token
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return None
    
    async def save_session(self, token: str):
        """Save session token to file"""
        try:
            session_data = {
                'token': token,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f)
                
            logger.info("Session saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
    
    async def get_session_token(self) -> str:
        """Get session token - either from file or generate new one"""
        # First try to load existing token
        token = await self.load_session()
        if token:
            return token
            
        if not self.totp_secret:
            # If no TOTP secret, we need to ask for a token
            login_url = f"https://api.icicidirect.com/apiuser/login?api_key={self.api_key}"
            logger.info(f"Please login at: {login_url}")
            logger.info("After login, copy the session token from the redirect URL")
            
            # In a real app, this would be handled by a UI component
            # For this implementation, we'll use a loop with timeout
            token = input("Enter session token: ")
            return token
        else:
            # Use TOTP for login
            totp = pyotp.TOTP(self.totp_secret)
            totp_code = totp.now()
            
            logger.info(f"Generated TOTP: {totp_code}")
            # Here we would automate the login process
            # But ICICI doesn't support fully automated login
            
            login_url = f"https://api.icicidirect.com/apiuser/login?api_key={self.api_key}"
            logger.info(f"Please login at: {login_url} using the TOTP: {totp_code}")
            
            token = input("Enter session token: ")
            return token

class ICICIDirectBroker(BaseBroker):
    """ICICI Direct broker implementation using Breeze Connect API"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ICICI Direct broker
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.totp_secret = config.get('totp_secret')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret are required")
            
        self.default_exchange = config.get('default_exchange', 'NSE')
        self.session_manager = SessionManager(
            self.api_key,
            self.api_secret,
            self.totp_secret
        )
        
        self.breeze = None
        self._connected = False
        
        # Rate limiting 
        self.request_count = 0
        self.request_limit = config.get('request_limit', 100)
        self.request_window = config.get('request_window', 60)  # in seconds
        self.last_request_reset = datetime.now()
        
        # Error handling
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2)  # in seconds
        
    async def connect(self) -> bool:
        """Connect to ICICI Direct API"""
        try:
            if self._connected and self.breeze:
                logger.info("Already connected to ICICI Direct")
                return True
                
            logger.info("Connecting to ICICI Direct...")
            
            # Initialize Breeze Connect
            self.breeze = BreezeConnect(api_key=self.api_key)
            
            # Get session token
            session_token = await self.session_manager.get_session_token()
            
            if not session_token:
                logger.error("Failed to get session token")
                return False
                
            # Generate session
            self.breeze.generate_session(
                api_secret=self.api_secret,
                session_token=session_token
            )
            
            # Save session token for future use
            await self.session_manager.save_session(session_token)
            
            # Test connection
            customer_details = self.breeze.get_customer_details()
            if 'Success' in customer_details:
                logger.info("Successfully connected to ICICI Direct")
                self._connected = True
                return True
            else:
                logger.error(f"Connection failed: {customer_details}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to ICICI Direct: {e}")
            return False
    
    async def is_connected(self) -> bool:
        """Check if connected to ICICI Direct API"""
        if not self._connected or not self.breeze:
            return False
            
        try:
            # Try a simple API call to check connection
            customer_details = self.breeze.get_customer_details()
            return 'Success' in customer_details
        except:
            self._connected = False
            return False
    
    async def _rate_limit_check(self):
        """Check and handle rate limiting"""
        current_time = datetime.now()
        
        # Reset counter if window has passed
        if (current_time - self.last_request_reset).seconds > self.request_window:
            self.request_count = 0
            self.last_request_reset = current_time
            
        # Check if we've exceeded the limit
        if self.request_count >= self.request_limit:
            wait_time = self.request_window - (current_time - self.last_request_reset).seconds
            logger.warning(f"Rate limit reached. Waiting {wait_time} seconds")
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.last_request_reset = datetime.now()
            
        self.request_count += 1
    
    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Check rate limits
                await self._rate_limit_check()
                
                # Check connection
                if not await self.is_connected():
                    logger.warning("Connection lost. Reconnecting...")
                    if not await self.connect():
                        raise ConnectionError("Failed to reconnect to ICICI Direct")
                
                # Execute the function
                return await func(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt+1}/{self.max_retries}: {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retrying with exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    # Last attempt failed, propagate the exception
                    raise
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current market quote for a symbol"""
        async def _get_quote():
            response = self.breeze.get_quotes(
                stock_code=symbol,
                exchange_code=self.default_exchange,
                product_type="cash",
                expiry_date="",
                strike_price="",
                right="",
                get_exchange_quotes=True,
                get_market_depth=True
            )
            
            if not response or 'Success' not in response or not response['Success']:
                logger.error(f"Failed to get quote for {symbol}")
                raise ValueError(f"Failed to get quote for {symbol}")
                
            quote_data = response['Success'][0]
            
            return {
                'symbol': symbol,
                'exchange': self.default_exchange,
                'ltp': float(quote_data.get('ltp', 0)),
                'change': float(quote_data.get('change', 0)),
                'change_percent': float(quote_data.get('change_percentage', 0)),
                'open': float(quote_data.get('open', 0)),
                'high': float(quote_data.get('high', 0)),
                'low': float(quote_data.get('low', 0)),
                'close': float(quote_data.get('close', 0)),
                'volume': int(quote_data.get('volume', 0)),
                'bid': float(quote_data.get('best_bid_price', 0)),
                'ask': float(quote_data.get('best_ask_price', 0)),
                'timestamp': datetime.now()
            }
            
        return await self._execute_with_retry(_get_quote)
    
    async def place_order(self, 
                        symbol: str,
                        quantity: int,
                        side: str,
                        order_type: str,
                        price: float = 0.0,
                        trigger_price: float = 0.0) -> Order:
        """Place a new order"""
        async def _place_order():
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
            
            # Convert to ICICI-specific format
            icici_side = "B" if side == "BUY" else "S"
            
            icici_order_type = {
                "MARKET": "MKT",
                "LIMIT": "L",
                "SL": "SL",
                "SL-M": "SL-M"
            }.get(order_type)
            
            # Place order
            response = self.breeze.place_order(
                stock_code=symbol,
                exchange_code=self.default_exchange,
                product="I",  # Intraday
                action=icici_side,
                quantity=str(quantity),
                order_type=icici_order_type,
                price=str(price) if price > 0 else "0",
                trigger_price=str(trigger_price) if trigger_price > 0 else "0",
                validity="DAY",
                disclosed_quantity="0",
                retention="DAY",
                remarks="GannTrading"
            )
            
            if not response or 'Success' not in response or not response['Success']:
                logger.error(f"Order failed: {response}")
                raise ValueError(f"Order placement failed: {response}")
                
            order_data = response['Success'][0]
            
            order = Order(
                order_id=order_data.get('order_id', ''),
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type=order_type,
                side=side,
                status=OrderStatus.PENDING
            )
            
            return order
            
        return await self._execute_with_retry(_place_order)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        async def _cancel_order():
            response = self.breeze.cancel_order(order_id=order_id)
            
            if not response or 'Success' not in response or not response['Success']:
                logger.error(f"Cancel order failed: {response}")
                return False
                
            return True
            
        return await self._execute_with_retry(_cancel_order)
    
    async def get_order_status(self, order_id: str) -> Order:
        """Get current status of an order"""
        async def _get_order_status():
            response = self.breeze.get_order_detail(
                exchange_code=self.default_exchange,
                order_id=order_id
            )
            
            if not response or 'Success' not in response or not response['Success']:
                logger.error(f"Get order status failed: {response}")
                raise ValueError(f"Failed to get order status: {response}")
                
            order_data = response['Success'][0]
            
            # Map ICICI status to our status
            status_map = {
                'Pending': OrderStatus.PENDING,
                'Complete': OrderStatus.COMPLETE,
                'Cancelled': OrderStatus.CANCELLED,
                'Rejected': OrderStatus.REJECTED
            }
            
            status = status_map.get(order_data.get('status', ''), OrderStatus.PENDING)
            
            order = Order(
                order_id=order_id,
                symbol=order_data.get('stock_code', ''),
                quantity=int(order_data.get('quantity', 0)),
                price=float(order_data.get('price', 0)),
                order_type=order_data.get('order_type', ''),
                side="BUY" if order_data.get('action', '') == 'B' else "SELL",
                status=status
            )
            
            # Add execution details if available
            if status == OrderStatus.COMPLETE:
                order.executed_price = float(order_data.get('average_price', 0))
                order.executed_quantity = int(order_data.get('filled_quantity', 0))
                order.execution_time = order_data.get('execution_time', '')
                
            return order
            
        return await self._execute_with_retry(_get_order_status)
    
    async def get_positions(self) -> List[Position]:
        """Get current open positions"""
        async def _get_positions():
            response = self.breeze.get_portfolio_positions()
            
            if not response or 'Success' not in response:
                logger.error(f"Get positions failed: {response}")
                raise ValueError(f"Failed to get positions: {response}")
                
            positions = []
            
            for pos_data in response['Success']:
                # Skip positions with 0 quantity
                quantity = int(pos_data.get('quantity', 0))
                if quantity == 0:
                    continue
                    
                side = "BUY" if quantity > 0 else "SELL"
                abs_quantity = abs(quantity)
                
                position = Position(
                    symbol=pos_data.get('stock_code', ''),
                    quantity=abs_quantity,
                    average_price=float(pos_data.get('average_price', 0)),
                    side=side
                )
                
                position.current_price = float(pos_data.get('last_price', 0))
                position.pnl = float(pos_data.get('pnl', 0))
                
                positions.append(position)
                
            return positions
            
        return await self._execute_with_retry(_get_positions)
    
    async def get_historical_data(self,
                                symbol: str,
                                timeframe: str,
                                start_date: datetime,
                                end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical price data"""
        async def _get_historical_data():
            # Convert timeframe to ICICI format
            interval = timeframe
            
            response = self.breeze.get_historical_data(
                interval=interval,
                from_date=start_date.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=end_date.strftime("%Y-%m-%d %H:%M:%S"),
                stock_code=symbol,
                exchange_code=self.default_exchange,
                product_type="cash"
            )
            
            if not response or 'Success' not in response:
                logger.error(f"Get historical data failed: {response}")
                raise ValueError(f"Failed to get historical data: {response}")
                
            candles = []
            
            for candle_data in response['Success']:
                candles.append({
                    'timestamp': candle_data.get('datetime', ''),
                    'open': float(candle_data.get('open', 0)),
                    'high': float(candle_data.get('high', 0)),
                    'low': float(candle_data.get('low', 0)),
                    'close': float(candle_data.get('close', 0)),
                    'volume': int(candle_data.get('volume', 0))
                })
                
            return candles
            
        return await self._execute_with_retry(_get_historical_data)
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance and margin information"""
        async def _get_account_balance():
            response = self.breeze.get_funds()
            
            if not response or 'Success' not in response:
                logger.error(f"Get account balance failed: {response}")
                raise ValueError(f"Failed to get account balance: {response}")
                
            balance_data = response['Success'][0]
            
            return {
                'total_balance': float(balance_data.get('limit', 0)),
                'used_margin': float(balance_data.get('utilized', 0)),
                'available_margin': float(balance_data.get('available', 0)),
                'cash_balance': float(balance_data.get('clear_balance', 0)),
                'collateral_margin': float(balance_data.get('collateral', 0))
            }
            
        return await self._execute_with_retry(_get_account_balance)
    
    async def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            return False
            
        # Check market hours (9:15 AM to 3:30 PM IST)
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    async def logout(self) -> bool:
        """Logout and clean up resources"""
        try:
            if self.breeze:
                # ICICI doesn't have an explicit logout method
                # but we can clear our local session
                self.breeze = None
                self._connected = False
                
            return True
            
        except Exception as e:
            logger.error(f"Error logging out: {e}")
            return False
