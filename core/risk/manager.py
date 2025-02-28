# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# core/risk/manager.py
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger("risk_manager")

class RiskLevel(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class RiskManager:
    """Risk management system to control trading risk"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager
        
        Args:
            config: Risk configuration parameters
        """
        self.config = config
        
        # Risk limits
        self.max_capital_per_trade = config.get('max_capital_per_trade', 0.02)  # 2% of capital
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5% of capital
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of capital
        self.max_positions = config.get('max_positions', 5)
        self.max_drawdown = config.get('max_drawdown', 0.1)  # 10% max drawdown
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_winning_trades = 0
        self.daily_losing_trades = 0
        
        # Position tracking
        self.positions = {}  # symbol -> position info
        self.capital_allocated = 0.0
        
        # Performance tracking
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.max_reached_drawdown = 0.0
        
        # Current risk level
        self.risk_level = RiskLevel.NORMAL
        
        # Timed reset
        self.last_reset = datetime.now().date()
        
        logger.info("Risk manager initialized")
    
    def check_daily_reset(self):
        """Check and perform daily reset if needed"""
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self._reset_daily_metrics()
            self.last_reset = current_date
    
    def _reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        logger.info("Resetting daily risk metrics")
        
        # Save metrics before reset
        daily_summary = {
            'date': self.last_reset,
            'pnl': self.daily_pnl,
            'trades': self.daily_trades,
            'winning_trades': self.daily_winning_trades,
            'losing_trades': self.daily_losing_trades,
            'win_rate': self.daily_winning_trades / self.daily_trades if self.daily_trades > 0 else 0,
            'drawdown': self.current_drawdown
        }
        
        # Reset metrics
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_winning_trades = 0
        self.daily_losing_trades = 0
        
        # Reset risk level if not critical
        if self.risk_level != RiskLevel.CRITICAL:
            self.risk_level = RiskLevel.NORMAL
            
        return daily_summary
    
    async def can_place_trade(self, 
                           symbol: str,
                           side: str,
                           quantity: int,
                           price: float) -> Dict[str, Any]:
        """
        Check if trade is allowed based on risk parameters
        
        Returns:
            Dict with 'allowed' (bool) and 'reason' (str) if not allowed
        """
        try:
            self.check_daily_reset()
            
            # Calculate trade value
            trade_value = quantity * price
            
            # Check if already have position
            has_position = symbol in self.positions
            
            if has_position and self.positions[symbol]['side'] == side:
                # Adding to position - check if it would exceed position limit
                new_position_value = self.positions[symbol]['value'] + trade_value
                if new_position_value > self.initial_capital * self.max_position_size:
                    return {
                        'allowed': False,
                        'reason': f"Position size would exceed {self.max_position_size:.1%} limit"
                    }
            elif not has_position:
                # New position - check number of positions
                if len(self.positions) >= self.max_positions:
                    return {
                        'allowed': False,
                        'reason': f"Maximum positions ({self.max_positions}) reached"
                    }
                
                # Check position size
                if trade_value > self.initial_capital * self.max_position_size:
                    return {
                        'allowed': False,
                        'reason': f"Position size exceeds {self.max_position_size:.1%} limit"
                    }
            
            # Check risk level
            if self.risk_level == RiskLevel.CRITICAL:
                return {
                    'allowed': False,
                    'reason': "Risk level critical - trading suspended"
                }
                
            # If risk level is warning, restrict to only closing trades
            if self.risk_level == RiskLevel.WARNING and not has_position:
                return {
                    'allowed': False,
                    'reason': "Risk level warning - only closing trades allowed"
                }
                
            # Check daily loss limit
            if self.daily_pnl <= -self.initial_capital * self.max_daily_loss:
                return {
                    'allowed': False,
                    'reason': "Daily loss limit reached"
                }
                
            # Check drawdown limit
            if self.current_drawdown >= self.max_drawdown:
                return {
                    'allowed': False,
                    'reason': "Maximum drawdown reached"
                }
                
            # All checks passed
            return {
                'allowed': True
            }
            
        except Exception as e:
            logger.error(f"Error in can_place_trade: {e}")
            return {
                'allowed': False,
                'reason': f"Risk check error: {str(e)}"
            }
    
    async def record_position_opened(self,
                                   symbol: str,
                                   side: str,
                                   quantity: int,
                                   price: float,
                                   order_id: str):
        """Record a new position being opened"""
        try:
            position_value = quantity * price
            
            # Record the position
            self.positions[symbol] = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'value': position_value,
                'order_id': order_id,
                'open_time': datetime.now()
            }
            
            # Update capital allocation
            self.capital_allocated += position_value
            
            logger.info(f"Position opened: {symbol} {side} {quantity} @ {price}")
            
        except Exception as e:
            logger.error(f"Error recording position: {e}")
    
    async def record_position_closed(self,
                                   symbol: str,
                                   exit_price: float,
                                   pnl: float):
        """Record a position being closed"""
        try:
            if symbol not in self.positions:
                logger.warning(f"Position not found for {symbol}")
                return
                
            position = self.positions[symbol]
            
            # Record trade result
            self.daily_trades += 1
            self.daily_pnl += pnl
            
            if pnl > 0:
                self.daily_winning_trades += 1
            else:
                self.daily_losing_trades += 1
                
            # Update capital based on P&L
            new_capital = self.initial_capital + self.daily_pnl
            
            # Update peak capital and drawdown
            if new_capital > self.peak_capital:
                self.peak_capital = new_capital
            else:
                self.current_drawdown = (self.peak_capital - new_capital) / self.peak_capital
                if self.current_drawdown > self.max_reached_drawdown:
                    self.max_reached_drawdown = self.current_drawdown
                    
            # Update capital allocation
            self.capital_allocated -= position['value']
            
            # Remove position
            del self.positions[symbol]
            
            # Update risk level based on metrics
            self._update_risk_level()
            
            logger.info(f"Position closed: {symbol} @ {exit_price}, P&L: {pnl}")
            
        except Exception as e:
            logger.error(f"Error recording position closure: {e}")
    
    async def record_position_updated(self,
                                    symbol: str,
                                    current_price: float,
                                    unrealized_pnl: float):
        """Update an existing position with new data"""
        try:
            if symbol not in self.positions:
                return
                
            position = self.positions[symbol]
            
            # Update position metrics
            position['current_price'] = current_price
            position['unrealized_pnl'] = unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def _update_risk_level(self):
        """Update risk level based on current metrics"""
        try:
            # Check drawdown
            if self.current_drawdown >= self.max_drawdown:
                self.risk_level = RiskLevel.CRITICAL
                logger.warning(f"Risk level set to CRITICAL due to drawdown: {self.current_drawdown:.2%}")
                return
                
            # Check daily loss
            daily_loss_pct = abs(self.daily_pnl) / self.initial_capital if self.daily_pnl < 0 else 0
            if daily_loss_pct >= self.max_daily_loss:
                self.risk_level = RiskLevel.CRITICAL
                logger.warning(f"Risk level set to CRITICAL due to daily loss: {daily_loss_pct:.2%}")
                return
                
            # Warning thresholds (80% of limits)
            if self.current_drawdown >= self.max_drawdown * 0.8:
                self.risk_level = RiskLevel.WARNING
                logger.warning(f"Risk level set to WARNING due to drawdown: {self.current_drawdown:.2%}")
                return
                
            if daily_loss_pct >= self.max_daily_loss * 0.8:
                self.risk_level = RiskLevel.WARNING
                logger.warning(f"Risk level set to WARNING due to daily loss: {daily_loss_pct:.2%}")
                return
                
            # Normal risk level
            self.risk_level = RiskLevel.NORMAL
            
        except Exception as e:
            logger.error(f"Error updating risk level: {e}")
    
    def calculate_position_size(self,
                              symbol: str,
                              price: float,
                              stop_loss: float) -> int:
        """
        Calculate appropriate position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Recommended quantity
        """
        try:
            # Risk per trade
            risk_per_trade_amount = self.initial_capital * self.max_capital_per_trade
            
            # Risk per unit
            risk_per_unit = abs(price - stop_loss)
            
            if risk_per_unit <= 0:
                logger.warning(f"Invalid risk per unit for {symbol}: {risk_per_unit}")
                return 0
                
            # Calculate quantity
            quantity = int(risk_per_trade_amount / risk_per_unit)
            
            # Apply position size limits
            max_position_value = self.initial_capital * self.max_position_size
            max_quantity = int(max_position_value / price)
            
            quantity = min(quantity, max_quantity)
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return {
            'risk_level': self.risk_level.value,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.initial_capital,
            'daily_trades': self.daily_trades,
            'daily_win_rate': self.daily_winning_trades / self.daily_trades if self.daily_trades > 0 else 0,
            'current_drawdown': self.current_drawdown,
            'max_reached_drawdown': self.max_reached_drawdown,
            'peak_capital': self.peak_capital,
            'capital_allocated': self.capital_allocated,
            'capital_allocated_pct': self.capital_allocated / self.initial_capital,
            'positions_count': len(self.positions),
            'last_reset': self.last_reset
        }

