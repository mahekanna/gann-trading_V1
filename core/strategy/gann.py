# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# core/strategy/gann.py
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base import BaseStrategy, Signal, SignalType
from ..brokers.base import BaseBroker

class GannSquareStrategy(BaseStrategy):
    """Gann Square of Nine Trading Strategy"""
    
    def __init__(self, broker: BaseBroker, config: Dict[str, Any]):
        """
        Initialize Gann Square strategy
        
        Args:
            broker: Broker for order execution
            config: Strategy configuration
        """
        super().__init__(broker, config)
        
        # Gann specific parameters
        self.gann_config = {
            'increments': config.get('gann_increments', 
                                   [0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]),
            'num_values': config.get('num_values', 35),
            'buffer_percentage': config.get('buffer_percentage', 0.002)
        }
        
        # Initialize trackers
        self.gann_levels = {}  # symbol -> levels
        self.last_close = {}   # symbol -> last close
        
        self.logger.info("Gann Square strategy initialized")
    
    async def generate_signal(self, symbol: str, quote: Dict[str, Any]) -> Optional[Signal]:
        """Generate Gann-based trading signal"""
        try:
            current_price = quote['ltp']
            close_price = quote.get('close', current_price)
            
            # Check if we have a significant price change to recalculate levels
            if symbol in self.last_close:
                price_change_pct = abs(close_price - self.last_close[symbol]) / self.last_close[symbol]
                significant_change = price_change_pct > self.gann_config['buffer_percentage']
            else:
                significant_change = True
                
            # Recalculate Gann levels if needed
            if significant_change or symbol not in self.gann_levels:
                self.last_close[symbol] = close_price
                gann_levels = self.calculate_gann_levels(close_price)
                self.gann_levels[symbol] = gann_levels
                
                # Log for debugging
                self.logger.debug(f"Recalculated Gann levels for {symbol}")
                self.logger.debug(f"Buy level: {gann_levels['buy_level']}")
                self.logger.debug(f"Sell level: {gann_levels['sell_level']}")
            else:
                gann_levels = self.gann_levels[symbol]
            
            # Check for signals
            if current_price >= gann_levels['buy_level']:
                return self._create_long_signal(symbol, current_price, gann_levels)
            elif current_price <= gann_levels['sell_level']:
                return self._create_short_signal(symbol, current_price, gann_levels)
            
            # Check if we need to exit existing position
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Check for trend reversal exit
                if position['side'] == "BUY" and current_price <= gann_levels['sell_level']:
                    return self._create_exit_signal(symbol, current_price)
                elif position['side'] == "SELL" and current_price >= gann_levels['buy_level']:
                    return self._create_exit_signal(symbol, current_price)
            
            # No signal
            return Signal(
                signal_type=SignalType.NO_SIGNAL,
                symbol=symbol,
                entry_price=0,
                stop_loss=0,
                targets=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def calculate_gann_levels(self, price: float) -> Dict[str, Any]:
        """Calculate Gann Square of 9 levels"""
        try:
            # Calculate Gann Square values
            gann_values = self.gann_square_of_9(
                price,
                self.gann_config['increments'],
                self.gann_config['num_values']
            )
            
            # Find nearest buy/sell levels
            buy_above, sell_below = self.find_buy_sell_levels(price, gann_values)
            
            # Get targets based on angles
            buy_targets, sell_targets = self.get_targets(
                price,
                gann_values,
                self.config.get('num_targets', 3)
            )
            
            # Calculate stop losses
            long_sl = buy_above['level'] - (buy_above['level'] - price) * 1.5
            short_sl = sell_below['level'] + (price - sell_below['level']) * 1.5
            
            # Ensure stops are beyond the recent price
            buffer = price * self.gann_config['buffer_percentage']
            long_sl = min(long_sl, price - buffer)
            short_sl = max(short_sl, price + buffer)
            
            return {
                'gann_values': gann_values,
                'buy_level': buy_above['level'],
                'sell_level': sell_below['level'],
                'buy_targets': buy_targets,
                'sell_targets': sell_targets,
                'long_stoploss': long_sl,
                'short_stoploss': short_sl
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Gann levels: {e}")
            return {
                'buy_level': 0,
                'sell_level': 0,
                'buy_targets': [],
                'sell_targets': [],
                'long_stoploss': 0,
                'short_stoploss': 0
            }
    
    def gann_square_of_9(self, 
                        price: float, 
                        increments: List[float],
                        num_values: int) -> Dict[str, List[float]]:
        """Calculate Gann Square of 9 values"""
        # Square root of base price
        root = math.sqrt(price)
        base = math.floor(root)
        
        # Define Gann angles
        angles = [
            "0", "45", "90", "135", "180", "225", "270", "315"
        ]
        
        # Calculate values for each angle
        gann_values = {}
        
        for angle, increment in zip(angles, increments):
            values = []
            for i in range(num_values):
                val = base + (i * increment)
                squared = val * val
                values.append(round(squared, 2))
            gann_values[angle] = values
        
        return gann_values
    
    def find_buy_sell_levels(self, price: float, gann_values: Dict[str, List[float]]) -> tuple:
        """Find nearest buy and sell levels"""
        # Find nearest level above current price
        levels_above = []
        levels_below = []
        
        for angle, values in gann_values.items():
            # Find values above current price
            above = [v for v in values if v > price]
            if above:
                nearest_above = min(above)
                levels_above.append({'angle': angle, 'level': nearest_above})
            
            # Find values below current price
            below = [v for v in values if v < price]
            if below:
                nearest_below = max(below)
                levels_below.append({'angle': angle, 'level': nearest_below})
        
        # Get the nearest level above and below
        if levels_above:
            buy_above = min(levels_above, key=lambda x: x['level'])
        else:
            buy_above = {'angle': '0', 'level': price * 1.01}
            
        if levels_below:
            sell_below = max(levels_below, key=lambda x: x['level'])
        else:
            sell_below = {'angle': '0', 'level': price * 0.99}
            
        return buy_above, sell_below
    
    def get_targets(self, 
                  price: float, 
                  gann_values: Dict[str, List[float]], 
                  num_targets: int) -> tuple:
        """Get target levels for both buy and sell signals"""
        # For buy signals, we want levels above the buy level
        # For sell signals, we want levels below the sell level
        
        # Flatten all values
        all_values = []
        for values in gann_values.values():
            all_values.extend(values)
            
        # Remove duplicates and sort
        all_values = sorted(set(all_values))
        
        # Find current price index
        try:
            price_index = next(i for i, v in enumerate(all_values) if v > price)
        except:
            price_index = len(all_values) - 1
            
        # Get buy targets (levels above price)
        buy_targets = []
        for i in range(1, num_targets+1):
            target_index = price_index + i
            if target_index < len(all_values):
                buy_targets.append(all_values[target_index])
            else:
                # If we run out of levels, create some
                buy_targets.append(price * (1 + 0.01 * i))
                
        # Get sell targets (levels below price)
        sell_targets = []
        for i in range(1, num_targets+1):
            target_index = price_index - i
            if target_index >= 0:
                sell_targets.append(all_values[target_index])
            else:
                # If we run out of levels, create some
                sell_targets.append(price * (1 - 0.01 * i))
                
        return buy_targets, sell_targets
    
    def _create_long_signal(self, 
                          symbol: str, 
                          price: float, 
                          gann_levels: Dict[str, Any]) -> Signal:
        """Create long signal from Gann levels"""
        return Signal(
            signal_type=SignalType.BUY,
            symbol=symbol,
            entry_price=price,
            stop_loss=gann_levels['long_stoploss'],
            targets=gann_levels['buy_targets']
        )
    
    def _create_short_signal(self, 
                           symbol: str, 
                           price: float, 
                           gann_levels: Dict[str, Any]) -> Signal:
        """Create short signal from Gann levels"""
        return Signal(
            signal_type=SignalType.SELL,
            symbol=symbol,
            entry_price=price,
            stop_loss=gann_levels['short_stoploss'],
            targets=gann_levels['sell_targets']
        )
    
    def _create_exit_signal(self, 
                          symbol: str, 
                          price: float) -> Signal:
        """Create exit signal"""
        return Signal(
            signal_type=SignalType.EXIT,
            symbol=symbol,
            entry_price=price,
            stop_loss=0,
            targets=[]
        )