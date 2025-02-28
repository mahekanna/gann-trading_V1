# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# core/engine/backtest.py
import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..brokers.base import BaseBroker, Order, Position, OrderStatus
from ..strategy.base import BaseStrategy, Signal, SignalType
from ..utils.logger import get_logger

logger = get_logger("backtest")

class BacktestBroker(BaseBroker):
    """Broker implementation for backtesting"""
    
    def __init__(self, historical_data: Dict[str, pd.DataFrame], config: Dict[str, Any]):
        """
        Initialize backtest broker
        
        Args:
            historical_data: Dictionary of historical price data (symbol -> DataFrame)
            config: Configuration parameters
        """
        self.data = historical_data
        self.config = config
        
        # Current simulation time
        self.current_time = None
        self.current_bar = {}  # symbol -> current bar index
        
        # Simulation parameters
        self.slippage = config.get('slippage', 0.0002)  # 0.02%
        self.commission = config.get('commission', 0.0003)  # 0.03%
        
        # Trading storage
        self.orders = {}     # order_id -> Order
        self.positions = {}  # symbol -> Position
        self.order_history = []
        self.trade_history = []
        
        # Account
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.equity = self.initial_capital
        self.cash = self.initial_capital
        
        # Order counter
        self.order_counter = 1
        
        # Connection status
        self._connected = True
        
        logger.info("Backtest broker initialized")
    
    async def connect(self) -> bool:
        """Connect to backtest broker (always succeeds)"""
        self._connected = True
        return True
    
    async def is_connected(self) -> bool:
        """Check if connected to backtest broker"""
        return self._connected
    
    async def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get current market quote for a symbol"""
        if symbol not in self.data:
            raise ValueError(f"No data for symbol: {symbol}")
            
        if self.current_time is None:
            raise ValueError("Backtest not started")
            
        # Find the bar for current time
        df = self.data[symbol]
        idx = self.current_bar.get(symbol, 0)
        
        if idx >= len(df):
            raise ValueError(f"No more data for {symbol}")
            
        bar = df.iloc[idx]
        
        return {
            'symbol': symbol,
            'ltp': bar['close'],
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume'],
            'timestamp': bar.name
        }
    
    async def place_order(self, 
                        symbol: str,
                        quantity: int,
                        side: str,
                        order_type: str,
                        price: float = 0.0,
                        trigger_price: float = 0.0) -> Order:
        """Place a new order in backtest"""
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
            
            # Create order ID
            order_id = f"BT_{self.order_counter:06d}"
            self.order_counter += 1
            
            # Create order with pending status
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
            
            # Process order immediately in backtest
            await self._process_order(order_id)
            
            return order
            
        except Exception as e:
            logger.error(f"Error placing backtest order: {e}")
            raise
    
    async def _process_order(self, order_id: str):
        """Process a pending order"""
        try:
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
                
                # Check if enough capital for BUY
                order_value = execution_price * order.quantity
                
                if order.side == "BUY" and order_value + transaction_cost > self.cash:
                    # Reject order due to insufficient funds
                    order.status = OrderStatus.REJECTED
                    order.execution_time = self.current_time
                    logger.warning(f"Order {order_id} rejected: Insufficient funds")
                    return
                
                # Execute order
                order.status = OrderStatus.COMPLETE
                order.executed_price = execution_price
                order.executed_quantity = order.quantity
                order.execution_time = self.current_time
                
                # Update capital
                if order.side == "BUY":
                    self.cash -= (order_value + transaction_cost)
                else:
                    self.cash += (order_value - transaction_cost)
                
                # Update position
                await self._update_position(order)
                
                # Add to history
                self.order_history.append(order)
                
                logger.debug(f"Order {order_id} executed at {execution_price}")
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
                # Adding to long position or reducing short position
                if position.side == "BUY":
                    # Adding to long position
                    new_quantity = position.quantity + quantity
                    new_avg_price = ((position.quantity * position.average_price) + 
                                   (quantity * price)) / new_quantity
                    
                    position.quantity = new_quantity
                    position.average_price = new_avg_price
                else:
                    # Reducing short position
                    if quantity >= position.quantity:
                        # Close short position and maybe create long position
                        
                        # Calculate profit/loss
                        pnl = (position.average_price - price) * position.quantity
                        
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
                        
                        # Create new long position with remaining quantity
                        remaining = quantity - position.quantity
                        
                        if remaining > 0:
                            # New long position
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                quantity=remaining,
                                average_price=price,
                                side="BUY"
                            )
                            self.positions[symbol].timestamp = order.execution_time
                        else:
                            # Position closed
                            del self.positions[symbol]
                    else:
                        # Partial close of short position
                        
                        # Calculate profit/loss
                        pnl = (position.average_price - price) * quantity
                        
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
            else:  # SELL
                # Adding to short position or reducing long position
                if position.side == "SELL":
                    # Adding to short position
                    new_quantity = position.quantity + quantity
                    new_avg_price = ((position.quantity * position.average_price) + 
                                   (quantity * price)) / new_quantity
                    
                    position.quantity = new_quantity
                    position.average_price = new_avg_price
                else:
                    # Reducing long position
                    if quantity >= position.quantity:
                        # Close long position and maybe create short position
                        
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
                        
                        # Create new short position with remaining quantity
                        remaining = quantity - position.quantity
                        
                        if remaining > 0:
                            # New short position
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                quantity=remaining,
                                average_price=price,
                                side="SELL"
                            )
                            self.positions[symbol].timestamp = order.execution_time
                        else:
                            # Position closed
                            del self.positions[symbol]
                    else:
                        # Partial close of long position
                        
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
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                side=side
            )
            self.positions[symbol].timestamp = self.current_time
    
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
        self.order_history.append(order)
        
        logger.debug(f"Order {order_id} cancelled")
        return True
    
    async def get_order_status(self, order_id: str) -> Order:
        """Get current status of an order"""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
            
        return self.orders[order_id]
    
    async def get_positions(self) -> List[Position]:
        """Get current open positions"""
        return list(self.positions.values())
    
    async def get_historical_data(self,
                                symbol: str,
                                timeframe: str,
                                start_date: datetime,
                                end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical price data (subset of the already loaded data)"""
        if symbol not in self.data:
            raise ValueError(f"No data for symbol: {symbol}")
            
        df = self.data[symbol]
        
        # Filter by date range
        filtered = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Convert to list of dicts
        result = []
        for idx, row in filtered.iterrows():
            result.append({
                'timestamp': idx,
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            })
            
        return result
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance and margin information"""
        # Calculate portfolio value
        portfolio_value = 0.0
        for position in self.positions.values():
            # Get current price
            symbol = position.symbol
            if symbol in self.data:
                df = self.data[symbol]
                idx = self.current_bar.get(symbol, 0)
                
                if idx < len(df):
                    current_price = df.iloc[idx]['close']
                    
                    if position.side == "BUY":
                        portfolio_value += position.quantity * current_price
                    else:
                        portfolio_value += position.quantity * (2 * position.average_price - current_price)
        
        # Calculate equity
        self.equity = self.cash + portfolio_value
        
        return {
            'total_balance': self.initial_capital,
            'equity': self.equity,
            'cash': self.cash,
            'used_margin': portfolio_value,
            'available_margin': self.cash
        }
    
    async def is_market_open(self) -> bool:
        """Check if market is currently open (always true in backtest)"""
        return True
    
    async def logout(self) -> bool:
        """Logout and clean up resources"""
        self._connected = False
        return True
    
    def advance_time(self, new_time: datetime):
        """Advance simulation time to a new point"""
        self.current_time = new_time
        
        # Process any pending orders
        for order_id in list(self.orders.keys()):
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                asyncio.run(self._process_order(order_id))
        
        # Update equity curve
        asyncio.run(self.get_account_balance())
    
    def advance_bar(self, symbol: str):
        """Advance to the next bar for a symbol"""
        if symbol in self.current_bar:
            self.current_bar[symbol] += 1
        else:
            self.current_bar[symbol] = 1
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get account equity history"""
        return pd.DataFrame({
            'equity': self.equity_curve
        })

class BacktestEngine:
    """Engine for backtesting trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtest engine
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.historical_data = {}
        self.broker = None
        self.strategies = []
        
        # Equity curve and trade history for analysis
        self.equity_curve = []
        self.trade_history = []
        
        logger.info("Backtest engine initialized")
    
    async def load_data(self, symbols: List[str], start_date: datetime, end_date: datetime, timeframe: str):
        """Load historical data for backtest"""
        try:
            logger.info(f"Loading historical data for {len(symbols)} symbols")
            
            for symbol in symbols:
                # In a real implementation, this would load from a data provider or database
                # Here we'll use a placeholder that would be replaced with actual data
                logger.info(f"Loading data for {symbol}")
                
                # Placeholder data generation
                dates = pd.date_range(start=start_date, end=end_date, freq=timeframe)
                data = pd.DataFrame({
                    'open': np.random.normal(100, 5, size=len(dates)),
                    'high': np.random.normal(105, 5, size=len(dates)),
                    'low': np.random.normal(95, 5, size=len(dates)),
                    'close': np.random.normal(100, 5, size=len(dates)),
                    'volume': np.random.randint(1000, 100000, size=len(dates))
                }, index=dates)
                
                # Ensure high, low, open, close are consistent
                data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 1, size=len(dates)))
                data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 1, size=len(dates)))
                
                self.historical_data[symbol] = data
                
            logger.info("Data loading completed")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def add_strategy(self, strategy_class, strategy_config: Dict[str, Any]):
        """Add a strategy to the backtest"""
        self.strategies.append((strategy_class, strategy_config))
        logger.info(f"Added strategy: {strategy_class.__name__}")
    
    async def run(self) -> Dict[str, Any]:
        """Run the backtest"""
        try:
            logger.info("Starting backtest")
            
            # Initialize broker with historical data
            self.broker = BacktestBroker(self.historical_data, self.config)
            
            # Initialize strategies
            initialized_strategies = []
            for strategy_class, strategy_config in self.strategies:
                strategy = strategy_class(self.broker, strategy_config)
                await strategy.initialize()
                initialized_strategies.append(strategy)
            
            # Get the earliest start date among all symbols
            start_date = min(data.index[0] for data in self.historical_data.values())
            
            # Get the latest end date among all symbols
            end_date = max(data.index[-1] for data in self.historical_data.values())
            
            logger.info(f"Backtest period: {start_date} to {end_date}")
            
            # Run each bar
            current_date = start_date
            bar_count = 0
            
            # Start strategies
            for strategy in initialized_strategies:
                await strategy.start()
                
            # Process each time bar
            while current_date <= end_date:
                # Update broker's current time
                self.broker.advance_time(current_date)
                
                # Process each symbol
                for symbol, data in self.historical_data.items():
                    if current_date in data.index:
                        # Advance to this bar
                        bar_idx = data.index.get_loc(current_date)
                        self.broker.current_bar[symbol] = bar_idx
                
                # Get account balance for equity curve
                account = await self.broker.get_account_balance()
                self.equity_curve.append({
                    'timestamp': current_date,
                    'equity': account['equity']
                })
                
                # Move to next bar
                bar_count += 1
                if bar_count % 100 == 0:
                    logger.info(f"Processed {bar_count} bars, current date: {current_date}")
                
                # Move to next date
                current_date += pd.Timedelta(data.index[1] - data.index[0])
            
            # Stop strategies
            for strategy in initialized_strategies:
                await strategy.stop()
                
            # Collect results
            results = self._collect_results()
            
            logger.info("Backtest completed")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _collect_results(self) -> Dict[str, Any]:
        """Collect and analyze backtest results"""
        try:
            # Get final equity and trade history
            equity_df = pd.DataFrame(self.equity_curve)
            trades_df = pd.DataFrame(self.broker.trade_history)
            
            # Calculate metrics
            initial_capital = self.config.get('initial_capital', 100000.0)
            final_equity = equity_df['equity'].iloc[-1] if not equity_df.empty else initial_capital
            
            # Return metrics
            return_pct = (final_equity - initial_capital) / initial_capital * 100
            
            # Drawdown
            equity_df['prev_peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['prev_peak']) / equity_df['prev_peak'] * 100
            max_drawdown = abs(equity_df['drawdown'].min())
            
            # Win rate
            if not trades_df.empty:
                win_rate = (trades_df['pnl'] > 0).mean() * 100
                profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                                  trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
            else:
                win_rate = 0
                profit_factor = 0
            
            # Results
            results = {
                'initial_capital': initial_capital,
                'final_equity': final_equity,
                'return_pct': return_pct,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(trades_df),
                'equity_curve': equity_df.to_dict(orient='records'),
                'trades': trades_df.to_dict(orient='records')
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error collecting results: {e}")
            return {}