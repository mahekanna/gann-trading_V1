# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:52:20 2025
@author: Mahesh Naidu
"""

# ui/terminal/app.py
import asyncio
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Static, Button, Input, Label,
    DataTable, OptionList, Checkbox, Switch, TabbedContent, TabPane
)
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.screen import Screen

from core.brokers.icici import ICICIDirectBroker
from core.brokers.paper import PaperBroker
from core.strategy.gann import GannSquareStrategy
from core.engine.execution import TradingEngine
from core.utils.logger import get_logger

logger = get_logger("terminal_ui")

class LoginScreen(Screen):
    """ICICI login screen"""
    
    def compose(self) -> ComposeResult:
        """Create screen widgets"""
        yield Header(show_clock=True)
        yield Container(
            Static("ICICI Direct Login", id="login-title", classes="title"),
            Static("Connect to ICICI Direct API", id="login-subtitle", classes="subtitle"),
            
            Static("Login URL:", id="login-url-label", classes="field-label"),
            Static("", id="login-url", classes="field-value"),
            
            Static("", id="totp-label", classes="field-label"),
            Static("", id="totp-value", classes="field-value"),
            
            Static("Session Token:", id="token-label", classes="field-label"),
            Input(placeholder="Enter session token from redirect URL", id="token-input", password=True),
            
            Static("Status:", id="status-label", classes="field-label"),
            Static("Not Connected", id="status-value", classes="field-value"),
            
            Button("Generate Login URL", id="gen-url-btn", variant="primary"),
            Button("Connect", id="connect-btn", variant="success"),
            Button("Test Connection", id="test-btn", variant="default"),
            Button("Continue to Trading", id="continue-btn", variant="warning", disabled=True),
            
            id="login-container"
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize screen"""
        # Check for API key
        api_key = os.environ.get("ICICI_API_KEY")
        if api_key:
            self.query_one("#gen-url-btn").disabled = False
        else:
            self.query_one("#gen-url-btn").disabled = True
            self.query_one("#login-url").update("[red]API key not found in environment[/red]")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "gen-url-btn":
            self.generate_login_url()
        elif button_id == "connect-btn":
            self.connect_to_icici()
        elif button_id == "test-btn":
            self.test_connection()
        elif button_id == "continue-btn":
            self.app.push_screen("trading")
    
    def generate_login_url(self) -> None:
        """Generate ICICI login URL"""
        api_key = os.environ.get("ICICI_API_KEY")
        if not api_key:
            self.notify("API key not found. Set ICICI_API_KEY environment variable.", severity="error")
            return
            
        login_url = f"https://api.icicidirect.com/apiuser/login?api_key={api_key}"
        self.query_one("#login-url").update(login_url)
        
        # Generate TOTP if available
        totp_secret = os.environ.get("ICICI_TOTP_SECRET")
        if totp_secret:
            try:
                import pyotp
                totp = pyotp.TOTP(totp_secret)
                totp_code = totp.now()
                
                self.query_one("#totp-label").update("TOTP Code:")
                self.query_one("#totp-value").update(f"[bold]{totp_code}[/bold]")
                
            except Exception as e:
                self.notify(f"Error generating TOTP: {e}", severity="error")
    
    def connect_to_icici(self) -> None:
        """Connect to ICICI API"""
        api_key = os.environ.get("ICICI_API_KEY")
        api_secret = os.environ.get("ICICI_API_SECRET")
        
        if not api_key or not api_secret:
            self.notify("API credentials not found. Set ICICI_API_KEY and ICICI_API_SECRET environment variables.", severity="error")
            return
            
        token = self.query_one("#token-input").value
        if not token:
            self.notify("Please enter session token", severity="warning")
            return
            
        # Connect
        self.query_one("#status-value").update("[yellow]Connecting...[/yellow]")
        
        # Run connection in a worker to avoid blocking UI
        self.run_worker(self._connect_worker(api_key, api_secret, token))
    
    async def _connect_worker(self, api_key: str, api_secret: str, token: str) -> None:
        """Worker to connect to ICICI API"""
        try:
            # Create broker config
            config = {
                'api_key': api_key,
                'api_secret': api_secret
            }
            
            # Create broker
            broker = ICICIDirectBroker(config)
            
            # Connect
            from breeze_connect import BreezeConnect
            
            breeze = BreezeConnect(api_key=api_key)
            breeze.generate_session(api_secret=api_secret, session_token=token)
            
            # Store in app
            self.app.broker = broker
            
            # Store raw breeze instance for direct access
            self.app.breeze_instance = breeze
            
            # Update UI
            self.query_one("#status-value").update("[green]Connected[/green]")
            self.query_one("#continue-btn").disabled = False
            
            self.notify("Successfully connected to ICICI Direct", severity="information")
            
        except Exception as e:
            self.query_one("#status-value").update(f"[red]Connection Failed: {str(e)}[/red]")
            self.notify(f"Connection failed: {e}", severity="error")
    
    def test_connection(self) -> None:
        """Test ICICI connection"""
        if not hasattr(self.app, 'breeze_instance'):
            self.notify("Not connected yet", severity="warning")
            return
            
        # Run test in a worker
        self.run_worker(self._test_worker())
    
    async def _test_worker(self) -> None:
        """Worker to test connection"""
        try:
            # Simple API call to test
            customer_details = self.app.breeze_instance.get_customer_details()
            
            if 'Success' in customer_details:
                self.notify("Connection test successful", severity="information")
            else:
                self.notify(f"Test failed: {customer_details}", severity="error")
                
        except Exception as e:
            self.notify(f"Test failed: {e}", severity="error")

class TradingScreen(Screen):
    """Main trading screen"""
    
    def compose(self) -> ComposeResult:
        """Create screen widgets"""
        yield Header(show_clock=True)
        yield Horizontal(
            # Left sidebar with controls
            Vertical(
                Static("Trading Controls", classes="title"),
                Button("Start Paper Trading", id="start-paper-btn", variant="primary"),
                Button("Start Live Trading", id="start-live-btn", variant="error", disabled=True),
                Button("Stop Trading", id="stop-btn", variant="warning", disabled=True),
                Static("Engine Status:", classes="field-label"),
                Static("Stopped", id="engine-status", classes="field-value"),
                Static("Mode:", classes="field-label"),
                Static("Paper", id="trading-mode", classes="field-value"),
                Static("Active Strategies:", classes="field-label"),
                Static("0", id="active-strategies", classes="field-value"),
                Static("", classes="spacer"),
                Static("Risk Settings", classes="subtitle"),
                Grid(
                    Label("Max Daily Loss %:"),
                    Input("5", id="max-daily-loss", classes="setting-input"),
                    Label("Max Position %:"),
                    Input("10", id="max-position-size", classes="setting-input"),
                    Label("Risk Per Trade %:"),
                    Input("2", id="risk-per-trade", classes="setting-input"),
                    id="risk-settings"
                ),
                id="sidebar"
            ),
            # Main content area with tabs
            TabbedContent(
                # Dashboard tab
                TabPane("Dashboard", id="dashboard-tab",
                    Grid(
                        Static("Account Balance", classes="panel-title"),
                        Static("₹0.00", id="account-balance", classes="panel-value"),
                        Static("Used Margin", classes="panel-title"),
                        Static("₹0.00", id="used-margin", classes="panel-value"),
                        Static("Available Margin", classes="panel-title"),
                        Static("₹0.00", id="available-margin", classes="panel-value"),
                        Static("Daily P&L", classes="panel-title"),
                        Static("₹0.00", id="daily-pnl", classes="panel-value"),
                        id="account-grid"
                    ),
                    Static("Active Positions", classes="section-title"),
                    self._create_positions_table(),
                ),
                # Trading tab
                TabPane("Trading", id="trading-tab",
                    Static("Order Entry", classes="section-title"),
                    Grid(
                        Label("Symbol:"),
                        Input(placeholder="e.g., SBIN", id="symbol-input"),
                        Label("Quantity:"),
                        Input(placeholder="e.g., 100", id="quantity-input"),
                        Label("Order Type:"),
                        OptionList("MARKET", "LIMIT", "SL", "SL-M", id="order-type"),
                        Label("Price:"),
                        Input(placeholder="Price (for LIMIT/SL)", id="price-input"),
                        id="order-entry-grid"
                    ),
                    Horizontal(
                        Button("BUY", id="buy-btn", variant="success"),
                        Button("SELL", id="sell-btn", variant="error"),
                        id="order-buttons"
                    ),
                    Static("Order Status:", classes="field-label"),
                    Static("", id="order-status", classes="field-value"),
                    Static("Recent Orders", classes="section-title"),
                    self._create_orders_table(),
                ),
                # Strategies tab
                TabPane("Strategies", id="strategies-tab",
                    Static("Gann Strategy Settings", classes="section-title"),
                    Grid(
                        Label("Symbols (comma separated):"),
                        Input("SBIN,RELIANCE,INFY", id="symbols-input"),
                        Label("Timeframe:"),
                        OptionList("1m", "5m", "15m", "30m", "60m", id="timeframe"),
                        Label("Gann Increments:"),
                        Input("0.125, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25", id="gann-increments"),
                        Label("Number of Values:"),
                        Input("35", id="num-values"),
                        Label("Buffer Percentage:"),
                        Input("0.002", id="buffer-percentage"),
                        id="strategy-settings"
                    ),
                    Button("Apply Settings", id="apply-settings-btn", variant="primary"),
                ),
                # Backtest tab
                TabPane("Backtest", id="backtest-tab",
                    Static("Backtesting", classes="section-title"),
                    Grid(
                        Label("Symbol:"),
                        Input("SBIN", id="backtest-symbol"),
                        Label("Start Date:"),
                        Input(placeholder="YYYY-MM-DD", id="start-date"),
                        Label("End Date:"),
                        Input(placeholder="YYYY-MM-DD", id="end-date"),
                        Label("Initial Capital:"),
                        Input("100000", id="initial-capital"),
                        id="backtest-settings"
                    ),
                    Button("Run Backtest", id="run-backtest-btn", variant="primary"),
                    Static("Backtest Results", classes="section-title"),
                    DataTable(id="backtest-results")
                ),
                id="main-content"
            ),
            id="trading-screen"
        )
        yield Footer()
    
    def _create_positions_table(self) -> DataTable:
        """Create positions table"""
        table = DataTable(id="positions-table")
        table.add_columns(
            "Symbol", "Side", "Quantity", "Entry Price", "Current Price", "P&L", "Status"
        )
        return table
    
    def _create_orders_table(self) -> DataTable:
        """Create orders table"""
        table = DataTable(id="orders-table")
        table.add_columns(
            "Time", "Symbol", "Side", "Type", "Quantity", "Price", "Status"
        )
        return table
    
    def on_mount(self) -> None:
        """Initialize screen"""
        # Setup refresh timer
        self.set_interval(1.0, self.refresh_data)
        
        # Initialize backtest results table
        backtest_table = self.query_one("#backtest-results")
        backtest_table.add_columns(
            "Metric", "Value"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "start-paper-btn":
            self.start_paper_trading()
        elif button_id == "start-live-btn":
            self.start_live_trading()
        elif button_id == "stop-btn":
            self.stop_trading()
        elif button_id == "buy-btn":
            self.place_order("BUY")
        elif button_id == "sell-btn":
            self.place_order("SELL")
        elif button_id == "apply-settings-btn":
            self.apply_strategy_settings()
        elif button_id == "run-backtest-btn":
            self.run_backtest()
    
    def start_paper_trading(self) -> None:
        """Start paper trading mode"""
        if not hasattr(self.app, 'broker'):
            self.notify("Not connected to broker", severity="error")
            return
            
        # Run in a worker
        self.run_worker(self._start_paper_worker())
    
    async def _start_paper_worker(self) -> None:
        """Worker to start paper trading"""
        try:
            # Update UI
            self.query_one("#engine-status").update("[yellow]Starting...[/yellow]")
            
            # Create paper broker
            live_broker = self.app.broker
            
            # Get risk settings
            max_daily_loss = float(self.query_one("#max-daily-loss").value) / 100
            max_position_size = float(self.query_one("#max-position-size").value) / 100
            risk_per_trade = float(self.query_one("#risk-per-trade").value) / 100
            
            # Create config
            config = {
                'paper_capital': 100000.0,
                'paper_slippage': 0.0002,
                'paper_commission': 0.0003,
                'max_daily_loss': max_daily_loss,
                'max_position_size': max_position_size,
                'risk_per_trade': risk_per_trade
            }
            
            # Create paper broker
            paper_broker = PaperBroker(live_broker, config)
            
            # Initialize paper broker
            await paper_broker.connect()
            
            # Create trading engine
            engine = TradingEngine(paper_broker, config)
            
            # Add strategies
            symbols_text = self.query_one("#symbols-input").value
            symbols = [s.strip() for s in symbols_text.split(',')]
            
            # Create strategy config
            strategy_config = {
                'symbols': symbols,
                'timeframe': self.query_one("#timeframe").value,
                'gann_increments': eval(f"[{self.query_one('#gann-increments').value}]"),
                'num_values': int(self.query_one("#num-values").value),
                'buffer_percentage': float(self.query_one("#buffer-percentage").value),
                'risk_per_trade': risk_per_trade,
                'account_balance': 100000.0
            }
            
            # Create strategy
            strategy = GannSquareStrategy(paper_broker, strategy_config)
            
            # Add to engine
            engine.add_strategy(strategy)
            
            # Initialize and start engine
            await engine.initialize()
            await engine.start()
            
            # Store in app
            self.app.trading_engine = engine
            self.app.trading_mode = "paper"
            
            # Update UI
            self.query_one("#engine-status").update("[green]Running[/green]")
            self.query_one("#trading-mode").update("Paper")
            self.query_one("#start-paper-btn").disabled = True
            self.query_one("#stop-btn").disabled = False
            
            self.notify("Paper trading started", severity="information")
            
        except Exception as e:
            self.query_one("#engine-status").update("[red]Error[/red]")
            self.notify(f"Error starting paper trading: {e}", severity="error")
    
    def start_live_trading(self) -> None:
        """Start live trading mode"""
        # Show warning dialog
        self.app.push_screen("live_warning")
    
    def stop_trading(self) -> None:
        """Stop trading"""
        if not hasattr(self.app, 'trading_engine'):
            return
            
        # Run in a worker
        self.run_worker(self._stop_trading_worker())
    
    async def _stop_trading_worker(self) -> None:
        """Worker to stop trading"""
        try:
            # Update UI
            self.query_one("#engine-status").update("[yellow]Stopping...[/yellow]")
            
            # Stop engine
            await self.app.trading_engine.stop()
            
            # Update UI
            self.query_one("#engine-status").update("[red]Stopped[/red]")
            self.query_one("#start-paper-btn").disabled = False
            self.query_one("#stop-btn").disabled = True
            
            self.notify("Trading stopped", severity="information")
            
        except Exception as e:
            self.notify(f"Error stopping trading: {e}", severity="error")
    
    def place_order(self, side: str) -> None:
        """Place a trading order"""
        if not hasattr(self.app, 'trading_engine'):
            self.notify("Trading engine not started", severity="error")
            return
            
        # Get order details
        symbol = self.query_one("#symbol-input").value
        if not symbol:
            self.notify("Please enter symbol", severity="warning")
            return
            
        quantity_text = self.query_one("#quantity-input").value
        if not quantity_text:
            self.notify("Please enter quantity", severity="warning")
            return
            
        try:
            quantity = int(quantity_text)
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
        except ValueError as e:
            self.notify(f"Invalid quantity: {e}", severity="error")
            return
            
        order_type = self.query_one("#order-type").value or "MARKET"
        
        price_text = self.query_one("#price-input").value
        price = 0.0
        if price_text:
            try:
                price = float(price_text)
                if price <= 0:
                    raise ValueError("Price must be positive")
            except ValueError as e:
                self.notify(f"Invalid price: {e}", severity="error")
                return
        
        # Run in a worker
        self.run_worker(self._place_order_worker(symbol, quantity, side, order_type, price))
    
    async def _place_order_worker(self, symbol: str, quantity: int, side: str,
                                 order_type: str, price: float) -> None:
        """Worker to place order"""
        try:
            # Update UI
            self.query_one("#order-status").update("[yellow]Placing order...[/yellow]")
            
            # Get broker
            broker = self.app.trading_engine.broker
            
            # Place order
            order = await broker.place_order(
                symbol=symbol,
                quantity=quantity,
                side=side,
                order_type=order_type,
                price=price
            )
            
            # Update UI
            if order.status == "COMPLETE":
                self.query_one("#order-status").update(f"[green]Order completed: {order.order_id}[/green]")
            else:
                self.query_one("#order-status").update(f"[yellow]Order {order.status}: {order.order_id}[/yellow]")
                
            self.notify(f"Order placed: {side} {quantity} {symbol}", severity="information")
            
        except Exception as e:
            self.query_one("#order-status").update(f"[red]Order error: {str(e)}[/red]")
            self.notify(f"Error placing order: {e}", severity="error")
    
    def apply_strategy_settings(self) -> None:
        """Apply strategy settings"""
        self.notify("Strategy settings applied", severity="information")
    
    def run_backtest(self) -> None:
        """Run backtest"""
        # Get backtest parameters
        symbol = self.query_one("#backtest-symbol").value
        start_date = self.query_one("#start-date").value
        end_date = self.query_one("#end-date").value
        initial_capital = self.query_one("#initial-capital").value
        
        # Validate
        if not all([symbol, start_date, end_date, initial_capital]):
            self.notify("Please fill all backtest parameters", severity="warning")
            return
            
        try:
            initial_capital = float(initial_capital)
            if initial_capital <= 0:
                raise ValueError("Initial capital must be positive")
                
            # Parse dates
            from datetime import datetime
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            if end <= start:
                raise ValueError("End date must be after start date")
                
        except ValueError as e:
            self.notify(f"Invalid parameter: {e}", severity="error")
            return
            
        # Run in a worker
        self.run_worker(self._run_backtest_worker(symbol, start, end, initial_capital))
    
    async def _run_backtest_worker(self, symbol: str, start_date: datetime,
                                 end_date: datetime, initial_capital: float) -> None:
        """Worker to run backtest"""
        try:
            from core.engine.backtest import BacktestEngine
            
            # Create config
            config = {
                'initial_capital': initial_capital,
                'slippage': 0.0002,
                'commission': 0.0003,
                'symbols': [symbol]
            }
            
            # Create backtest engine
            engine = BacktestEngine(config)
            
            # Load data
            await engine.load_data([symbol], start_date, end_date, "15m")
            
            # Get strategy settings
            strategy_config = {
                'symbols': [symbol],
                'timeframe': self.query_one("#timeframe").value,
                'gann_increments': eval(f"[{self.query_one('#gann-increments').value}]"),
                'num_values': int(self.query_one("#num-values").value),
                'buffer_percentage': float(self.query_one("#buffer-percentage").value),
                'risk_per_trade': float(self.query_one("#risk-per-trade").value) / 100,
                'account_balance': initial_capital
            }
            
            # Add strategy
            from core.strategy.gann import GannSquareStrategy
            engine.add_strategy(GannSquareStrategy, strategy_config)
            
            # Run backtest
            results = await engine.run()
            
            # Update UI with results
            self._display_backtest_results(results)
            
            self.notify("Backtest completed", severity="information")
            
        except Exception as e:
            self.notify(f"Error running backtest: {e}", severity="error")
    
    def _display_backtest_results(self, results: Dict[str, Any]) -> None:
        """Display backtest results in UI"""
        table = self.query_one("#backtest-results")
        table.clear()
        
        # Add rows
        table.add_row("Initial Capital", f"₹{results['initial_capital']:,.2f}")
        table.add_row("Final Equity", f"₹{results['final_equity']:,.2f}")
        table.add_row("Return", f"{results['return_pct']:,.2f}%")
        table.add_row("Max Drawdown", f"{results['max_drawdown']:,.2f}%")
        table.add_row("Win Rate", f"{results['win_rate']:,.2f}%")
        table.add_row("Profit Factor", f"{results['profit_factor']:,.2f}")
        table.add_row("Total Trades", str(results['total_trades']))
    
    async def refresh_data(self) -> None:
        """Refresh UI data"""
        if not hasattr(self.app, 'trading_engine'):
            return
            
        try:
            # Get engine status
            status = self.app.trading_engine.get_status()
            
            # Update status display
            self.query_one("#engine-status").update(f"[green]{status['state']}[/green]")
            self.query_one("#active-strategies").update(str(len(status['strategies'])))
            
            # Update positions table
            await self._refresh_positions()
            
            # Update orders table
            await self._refresh_orders()
            
            # Update account info
            await self._refresh_account_info()
            
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
    
    async def _refresh_positions(self) -> None:
        """Refresh positions table"""
        if not hasattr(self.app, 'trading_engine'):
            return
            
        try:
            # Get broker
            broker = self.app.trading_engine.broker
            
            # Get positions
            positions = await broker.get_positions()
            
            # Update table
            table = self.query_one("#positions-table")
            table.clear()
            
            for pos in positions:
                # Format P&L with color
                if pos.pnl >= 0:
                    pnl_text = f"[green]₹{pos.pnl:,.2f}[/green]"
                else:
                    pnl_text = f"[red]₹{pos.pnl:,.2f}[/red]"
                    
                table.add_row(
                    pos.symbol,
                    pos.side,
                    str(pos.quantity),
                    f"₹{pos.average_price:,.2f}",
                    f"₹{pos.current_price:,.2f}",
                    pnl_text,
                    "OPEN"
                )
                
        except Exception as e:
            logger.error(f"Error refreshing positions: {e}")
    
    async def _refresh_orders(self) -> None:
        """Refresh orders table"""
        if not hasattr(self.app, 'trading_engine'):
            return
            
        # In a real implementation, we would get orders from the broker
        # For now, we'll skip this since we don't have an order history API
        pass
    
    async def _refresh_account_info(self) -> None:
        """Refresh account information"""
        if not hasattr(self.app, 'trading_engine'):
            return
            
        try:
            # Get broker
            broker = self.app.trading_engine.broker
            
            # Get account balance
            account = await broker.get_account_balance()
            
            # Update display
            self.query_one("#account-balance").update(f"₹{account.get('total_balance', 0):,.2f}")
            self.query_one("#used-margin").update(f"₹{account.get('used_margin', 0):,.2f}")
            self.query_one("#available-margin").update(f"₹{account.get('available_margin', 0):,.2f}")
            
            # For paper trading, we can get daily P&L directly
            if hasattr(broker, 'get_trade_history'):
                trades = broker.get_trade_history()
                today = datetime.now().date()
                daily_pnl = sum(t.get('pnl', 0) for t in trades 
                              if t.get('exit_time', datetime.now()).date() == today)
                
                if daily_pnl >= 0:
                    self.query_one("#daily-pnl").update(f"[green]₹{daily_pnl:,.2f}[/green]")
                else:
                    self.query_one("#daily-pnl").update(f"[red]₹{daily_pnl:,.2f}[/red]")
                
        except Exception as e:
            logger.error(f"Error refreshing account info: {e}")

class LiveWarningScreen(Screen):
    """Warning screen for live trading"""
    
    def compose(self) -> ComposeResult:
        """Create screen widgets"""
        yield Static("⚠️ WARNING: LIVE TRADING ⚠️", id="warning-title", classes="warning-title")
        yield Static("You are about to start LIVE trading with REAL money!", 
                   id="warning-text", classes="warning-text")
        yield Static("This will place actual orders on the exchange using your account.", 
                   classes="warning-detail")
        yield Static("", classes="spacer")
        yield Checkbox("I understand the risks of live trading", id="risk-checkbox")
        yield Checkbox("I have tested my strategy in paper trading", id="test-checkbox")
        yield Checkbox("I accept full responsibility for any losses", id="responsibility-checkbox")
        yield Static("", classes="spacer")
        yield Grid(
            Button("Cancel", id="cancel-btn", variant="primary"),
            Button("Proceed to Live Trading", id="proceed-btn", variant="error", disabled=True),
            id="warning-buttons"
        )
    
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes"""
        # Enable proceed button only if all checkboxes are checked
        all_checked = all(
            self.query_one(f"#{cb_id}").value
            for cb_id in ["risk-checkbox", "test-checkbox", "responsibility-checkbox"]
        )
        
        self.query_one("#proceed-btn").disabled = not all_checked
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "cancel-btn":
            self.app.pop_screen()
        elif event.button.id == "proceed-btn":
            # Start live trading
            self.app.pop_screen()
            self.app.query_one(TradingScreen).run_worker(
                self.app.query_one(TradingScreen)._start_live_worker()
            )

class GannTradingApp(App):
    """Main trading application"""
    
    CSS = """
    /* Global styles */
    Screen {
        background: $surface;
    }
    
    .title {
        text-align: center;
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 1;
        margin-bottom: 1;
    }
    
    .subtitle {
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }
    
    .section-title {
        text-style: bold;
        background: $primary-darken-1;
        color: $text;
        padding: 1;
        margin-top: 1;
        margin-bottom: 1;
    }
    
    .field-label {
        margin-right: 1;
    }
    
    .field-value {
        margin-left: 1;
    }
    
    .spacer {
        height: 1;
    }
    
    /* Login screen */
    #login-container {
        width: 60;
        margin: 1 auto;
        border: solid $primary;
        padding: 1;
    }
    
    /* Trading screen */
    #trading-screen {
        width: 100%;
        height: 100%;
    }
    
    #sidebar {
        width: 25%;
        height: 100%;
        border-right: solid $primary;
        padding: 1;
    }
    
    #main-content {
        width: 75%;
        height: 100%;
        padding: 1;
    }
    
    /* Dashboard */
    #account-grid {
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: 4;
        height: auto;
        margin: 1;
    }
    
    .panel-title {
        text-style: bold;
    }
    
    .panel-value {
        text-align: right;
    }
    
    /* Tables */
    DataTable {
        height: 8;
        margin-bottom: 1;
    }
    
    /* Order entry */
    #order-entry-grid {
        grid-size: 2;
        grid-columns: 1fr 3fr;
        grid-rows: 4;
        height: auto;
        margin: 1;
    }
    
    #order-buttons {
        margin: 1;
        height: 3;
    }
    
    /* Strategy settings */
    #strategy-settings {
        grid-size: 2;
        grid-columns: 1fr 3fr;
        grid-rows: 5;
        height: auto;
        margin: 1;
    }
    
    #risk-settings {
        grid-size: 2;
        grid-columns: 2fr 1fr;
        grid-rows: 3;
        height: auto;
    }
    
    .setting-input {
        width: 100%;
    }
    
    /* Backtest */
    #backtest-settings {
        grid-size: 2;
        grid-columns: 1fr 3fr;
        grid-rows: 4;
        height: auto;
        margin: 1;
    }
    
    /* Live warning screen */
    #warning-title {
        text-align: center;
        text-style: bold;
        background: $error;
        color: $text;
        padding: 2;
        margin-bottom: 2;
    }
    
    .warning-text {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .warning-detail {
        text-align: center;
        margin-bottom: 1;
    }
    
    #warning-buttons {
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-rows: 1;
        height: auto;
        margin: 2;
    }
    """
    
    SCREENS = {
        "login": LoginScreen,
        "trading": TradingScreen,
        "live_warning": LiveWarningScreen
    }
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle Dark Mode")
    ]
    
    def __init__(self):
        super().__init__()
        # Will store broker and engine references
        self.broker = None
        self.breeze_instance = None
        self.trading_engine = None
        self.trading_mode = None
    
    def on_mount(self) -> None:
        """Initialize app"""
        self.push_screen("login")
        
    def action_toggle_dark(self) -> None:
        """Toggle dark mode"""
        self.dark = not self.dark