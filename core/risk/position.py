# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""


# core/risk/position.py
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..utils.logger import get_logger

logger = get_logger("position_tracker")

class PositionTracker:
    """Track and manage trading positions"""
    
    def __init__(self, risk_manager):
        """
        Initialize position tracker
        
        Args:
            risk_manager: Risk manager instance
        """
        self.risk_manager = risk_manager
        self.positions = {}  # symbol -> position info
        self.order_map = {}  # order_id -> symbol
        self.history = []    # closed positions
    
    async def add_position(self, 
                        symbol: str,
                        side: str,
                        quantity: int,
                        entry_price: float,
                        order_id: str,
                        stop_loss: Optional[float] = None,
                        targets: Optional[List[float]] = None):
        """Add a new trading position"""
        try:
            position_value = quantity * entry_price
            
            # Create position
            position = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'entry_order_id': order_id,
                'current_price': entry_price,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'value': position_value,
                'stop_loss': stop_loss,
                'targets': targets or [],
                'targets_hit': 0,
                'status': 'OPEN'
            }
            
            # Record in tracking maps
            self.positions[symbol] = position
            self.order_map[order_id] = symbol
            
            # Notify risk manager
            await self.risk_manager.record_position_opened(
                symbol, side, quantity, entry_price, order_id
            )
            
            logger.info(f"Position added: {symbol} {side} {quantity} @ {entry_price}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return None
    
    async def update_position(self, 
                           symbol: str,
                           current_price: float,
                           timestamp: Optional[datetime] = None):
        """Update position with current price"""
        try:
            if symbol not in self.positions:
                return
                
            position = self.positions[symbol]
            position['current_price'] = current_price
            
            # Calculate unrealized P&L
            if position['side'] == 'BUY':
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
            else:
                position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']
                
            # Update timestamp
            if timestamp:
                position['last_update'] = timestamp
            else:
                position['last_update'] = datetime.now()
                
            # Notify risk manager
            await self.risk_manager.record_position_updated(
                symbol, current_price, position['unrealized_pnl']
            )
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    async def close_position(self, 
                          symbol: str,
                          exit_price: float,
                          exit_order_id: str,
                          reason: str = 'User Exit'):
        """Close a trading position"""
        try:
            if symbol not in self.positions:
                logger.warning(f"Position not found for {symbol}")
                return None
                
            position = self.positions[symbol]
            
            # Calculate realized P&L
            if position['side'] == 'BUY':
                realized_pnl = (exit_price - position['entry_price']) * position['quantity']
            else:
                realized_pnl = (position['entry_price'] - exit_price) * position['quantity']
                
            # Update position
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['exit_order_id'] = exit_order_id
            position['realized_pnl'] = realized_pnl
            position['exit_reason'] = reason
            position['status'] = 'CLOSED'
            
            # Save to history
            self.history.append(position.copy())
            
            # Notify risk manager
            await self.risk_manager.record_position_closed(
                symbol, exit_price, realized_pnl
            )
            
            # Remove from active positions
            del self.positions[symbol]
            
            logger.info(f"Position closed: {symbol} @ {exit_price}, P&L: {realized_pnl}")
            
            return position
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    async def partial_close(self,
                          symbol: str,
                          quantity: int,
                          exit_price: float,
                          exit_order_id: str,
                          reason: str = 'Partial Exit'):
        """Partially close a position"""
        try:
            if symbol not in self.positions:
                logger.warning(f"Position not found for {symbol}")
                return None
                
            position = self.positions[symbol]
            
            # Ensure quantity is valid
            if quantity > position['quantity']:
                logger.warning(f"Partial close quantity ({quantity}) exceeds position quantity ({position['quantity']})")
                quantity = position['quantity']
                
            if quantity <= 0:
                logger.warning(f"Invalid partial close quantity: {quantity}")
                return None
                
            # Calculate realized P&L for this part
            if position['side'] == 'BUY':
                realized_pnl = (exit_price - position['entry_price']) * quantity
            else:
                realized_pnl = (position['entry_price'] - exit_price) * quantity
                
            # Create partial exit record
            partial_exit = {
                'symbol': symbol,
                'side': position['side'],
                'quantity': quantity,
                'entry_price': position['entry_price'],
                'entry_time': position['entry_time'],
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'realized_pnl': realized_pnl,
                'exit_reason': reason,
                'partial': True
            }
            
            # Update position
            position['quantity'] -= quantity
            position['realized_pnl'] += realized_pnl
            
            # If no quantity left, close position
            if position['quantity'] <= 0:
                # Save to history
                position['exit_price'] = exit_price
                position['exit_time'] = datetime.now()
                position['exit_order_id'] = exit_order_id
                position['exit_reason'] = reason
                position['status'] = 'CLOSED'
                
                self.history.append(position.copy())
                
                # Notify risk manager
                await self.risk_manager.record_position_closed(
                    symbol, exit_price, realized_pnl
                )
                
                # Remove from active positions
                del self.positions[symbol]
            else:
                # Update value
                position['value'] = position['quantity'] * position['entry_price']
                
                # Save partial exit to history
                self.history.append(partial_exit)
                
                # We still have a position, so just update risk manager
                unrealized_pnl = 0
                if position['side'] == 'BUY':
                    unrealized_pnl = (exit_price - position['entry_price']) * position['quantity']
                else:
                    unrealized_pnl = (position['entry_price'] - exit_price) * position['quantity']
                    
                await self.risk_manager.record_position_updated(
                    symbol, exit_price, unrealized_pnl
                )
            
            logger.info(f"Partial close: {symbol} {quantity} @ {exit_price}, P&L: {realized_pnl}")
            
            return partial_exit
            
        except Exception as e:
            logger.error(f"Error in partial close: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position by symbol"""
        return self.positions.get(symbol)
    
    def get_position_by_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get position by order ID"""
        symbol = self.order_map.get(order_id)
        if symbol:
            return self.positions.get(symbol)
        return None
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions"""
        return list(self.positions.values())
    
    def get_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get position history for specified number of days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [pos for pos in self.history if pos['exit_time'] >= cutoff]