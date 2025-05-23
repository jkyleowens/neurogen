#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Realistic Trading Simulation System for Brain-Inspired Neural Network

This module transforms the neural network into a trading agent that makes
actual buy/sell/hold decisions with position sizing, and evaluates performance
using realistic trading metrics including transaction costs, slippage, and
comprehensive financial analysis.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradingAction:
    """Represents a trading action with position sizing"""
    action: str  # 'buy', 'sell', 'hold'
    position_size: float  # Fraction of available capital (0.0 to 1.0)
    confidence: float  # Model confidence in the decision (0.0 to 1.0)
    expected_return: float  # Expected return from this action
    
class TradingEnvironment:
    """
    Realistic trading environment with transaction costs, slippage, and position management
    """
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005,  # 0.05% slippage
                 max_position_size: float = 0.2,  # Max 20% of capital per trade
                 min_trade_amount: float = 100.0):  # Minimum trade amount
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.min_trade_amount = min_trade_amount
        
        # Portfolio state
        self.cash = initial_capital
        self.shares = 0.0
        self.current_price = 0.0
        
        # Trading history
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []
        self.positions = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        
    def reset(self):
        """Reset the trading environment"""
        self.current_capital = self.initial_capital
        self.cash = self.initial_capital
        self.shares = 0.0
        self.current_price = 0.0
        self.trades.clear()
        self.portfolio_values.clear()
        self.daily_returns.clear()
        self.positions.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees = 0.0
        
    def get_portfolio_value(self, price: float) -> float:
        """Calculate current portfolio value"""
        return self.cash + self.shares * price
        
    def execute_trade(self, action: TradingAction, price: float, timestamp: str) -> Dict:
        """
        Execute a trading action and return trade details
        """
        self.current_price = price
        trade_details = {
            'timestamp': timestamp,
            'action': action.action,
            'price': price,
            'shares_before': self.shares,
            'cash_before': self.cash,
            'portfolio_value_before': self.get_portfolio_value(price),
            'confidence': action.confidence,
            'expected_return': action.expected_return
        }
        
        if action.action == 'hold':
            # No action taken
            trade_details.update({
                'shares_traded': 0,
                'trade_value': 0,
                'fees': 0,
                'shares_after': self.shares,
                'cash_after': self.cash,
                'portfolio_value_after': self.get_portfolio_value(price)
            })
            
        elif action.action == 'buy':
            # Calculate trade amount
            available_cash = self.cash
            max_trade_value = available_cash * min(action.position_size, self.max_position_size)
            
            if max_trade_value >= self.min_trade_amount:
                # Apply slippage (buy at slightly higher price)
                effective_price = price * (1 + self.slippage)
                
                # Calculate shares to buy
                gross_shares = max_trade_value / effective_price
                fees = max_trade_value * self.transaction_cost
                net_cash_needed = max_trade_value + fees
                
                if net_cash_needed <= self.cash:
                    # Execute buy order
                    shares_bought = gross_shares
                    self.shares += shares_bought
                    self.cash -= net_cash_needed
                    self.total_trades += 1
                    self.total_fees += fees
                    
                    trade_details.update({
                        'shares_traded': shares_bought,
                        'trade_value': max_trade_value,
                        'effective_price': effective_price,
                        'fees': fees,
                        'shares_after': self.shares,
                        'cash_after': self.cash,
                        'portfolio_value_after': self.get_portfolio_value(price)
                    })
                    
                    self.trades.append(trade_details.copy())
                else:
                    # Insufficient funds
                    trade_details['action'] = 'hold'
                    trade_details['reason'] = 'insufficient_funds'
            else:
                # Trade amount too small
                trade_details['action'] = 'hold'
                trade_details['reason'] = 'trade_too_small'
                
        elif action.action == 'sell':
            if self.shares > 0:
                # Calculate shares to sell
                shares_to_sell = min(self.shares, self.shares * action.position_size)
                
                if shares_to_sell > 0:
                    # Apply slippage (sell at slightly lower price)
                    effective_price = price * (1 - self.slippage)
                    
                    # Calculate trade value
                    gross_proceeds = shares_to_sell * effective_price
                    fees = gross_proceeds * self.transaction_cost
                    net_proceeds = gross_proceeds - fees
                    
                    # Execute sell order
                    self.shares -= shares_to_sell
                    self.cash += net_proceeds
                    self.total_trades += 1
                    self.total_fees += fees
                    
                    trade_details.update({
                        'shares_traded': -shares_to_sell,
                        'trade_value': gross_proceeds,
                        'effective_price': effective_price,
                        'fees': fees,
                        'shares_after': self.shares,
                        'cash_after': self.cash,
                        'portfolio_value_after': self.get_portfolio_value(price)
                    })
                    
                    self.trades.append(trade_details.copy())
                else:
                    # No shares to sell
                    trade_details['action'] = 'hold'
                    trade_details['reason'] = 'no_shares'
            else:
                # No shares to sell
                trade_details['action'] = 'hold'
                trade_details['reason'] = 'no_shares'
        
        # Update portfolio tracking
        portfolio_value = self.get_portfolio_value(price)
        self.portfolio_values.append(portfolio_value)
        
        # Calculate daily return
        if len(self.portfolio_values) > 1:
            daily_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.daily_returns.append(daily_return)
        
        # Track position
        self.positions.append({
            'timestamp': timestamp,
            'price': price,
            'shares': self.shares,
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'position_pct': (self.shares * price) / portfolio_value if portfolio_value > 0 else 0
        })
        
        return trade_details

class TechnicalIndicators:
    """
    Calculate comprehensive technical indicators for trading decisions
    """
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, periods: Dict = None) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if periods is None:
            periods = {
                'sma_short': 10, 'sma_long': 30, 'ema_short': 12, 'ema_long': 26,
                'rsi': 14, 'bb': 20, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9
            }
        
        result = df.copy()
        
        # Moving Averages
        result['SMA_10'] = df['Close'].rolling(window=periods['sma_short']).mean()
        result['SMA_30'] = df['Close'].rolling(window=periods['sma_long']).mean()
        result['EMA_12'] = df['Close'].ewm(span=periods['ema_short']).mean()
        result['EMA_26'] = df['Close'].ewm(span=periods['ema_long']).mean()
        
        # RSI
        result['RSI'] = TechnicalIndicators.calculate_rsi(df['Close'], periods['rsi'])
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = TechnicalIndicators.calculate_bollinger_bands(
            df['Close'], periods['bb'])
        result['BB_Upper'] = bb_upper
        result['BB_Lower'] = bb_lower
        result['BB_Middle'] = bb_middle
        result['BB_Width'] = (bb_upper - bb_lower) / bb_middle
        result['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        # MACD
        macd_line, macd_signal, macd_histogram = TechnicalIndicators.calculate_macd(
            df['Close'], periods['macd_fast'], periods['macd_slow'], periods['macd_signal'])
        result['MACD'] = macd_line
        result['MACD_Signal'] = macd_signal
        result['MACD_Histogram'] = macd_histogram
        
        # Volatility
        result['ATR'] = TechnicalIndicators.calculate_atr(df, 14)
        result['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Volume indicators
        if 'Volume' in df.columns:
            result['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            result['Volume_Ratio'] = df['Volume'] / result['Volume_MA']
            
            # On-Balance Volume
            result['OBV'] = TechnicalIndicators.calculate_obv(df['Close'], df['Volume'])
            
        # Price momentum
        result['Price_Change_1d'] = df['Close'].pct_change(1)
        result['Price_Change_5d'] = df['Close'].pct_change(5)
        result['Price_Change_10d'] = df['Close'].pct_change(10)
        
        # Support/Resistance levels
        result['Resistance_20'] = df['High'].rolling(window=20).max()
        result['Support_20'] = df['Low'].rolling(window=20).min()
        result['Price_Position'] = (df['Close'] - result['Support_20']) / (result['Resistance_20'] - result['Support_20'])
        
        # Market structure
        result['Higher_High'] = (df['High'] > df['High'].shift(1)).astype(int)
        result['Lower_Low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        
        return result.fillna(method='ffill').fillna(0)
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band, sma
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift())
        low_close_prev = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_obv(prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        price_change = prices.diff()
        direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        obv = (volume * direction).cumsum()
        return obv

class TradingAgent:
    """
    Neural network trading agent that makes buy/sell/hold decisions
    """
    def __init__(self, model, device, confidence_threshold: float = 0.6):
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Trading decision thresholds
        self.buy_threshold = 0.6  # Model output > 0.6 = buy signal
        self.sell_threshold = 0.4  # Model output < 0.4 = sell signal
        
    def process_market_data(self, market_data: pd.DataFrame) -> torch.Tensor:
        """
        Process market data into model input format
        """
        # Select relevant features for the model
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_30', 'EMA_12', 'EMA_26', 'RSI',
            'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'ATR', 'Volatility', 'Volume_Ratio',
            'Price_Change_1d', 'Price_Change_5d', 'Price_Position'
        ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in market_data.columns]
        
        if len(available_columns) < 5:
            # Fallback to basic OHLCV
            available_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Extract features
        features = market_data[available_columns].values
        
        # Normalize features
        features = self._normalize_features(features)
        
        # Convert to tensor
        tensor = torch.FloatTensor(features).to(self.device)
        
        # Add batch dimension if needed
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
            
        return tensor
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to reasonable ranges"""
        # Simple min-max normalization with outlier clipping
        normalized = features.copy()
        
        for i in range(features.shape[1]):
            column = features[:, i]
            # Clip outliers
            q1, q99 = np.percentile(column, [1, 99])
            column = np.clip(column, q1, q99)
            
            # Min-max normalize
            min_val, max_val = column.min(), column.max()
            if max_val > min_val:
                normalized[:, i] = (column - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.5  # Constant values
                
        return normalized
    
    def make_trading_decision(self, market_data: pd.DataFrame, current_position: float = 0.0) -> TradingAction:
        """
        Make a trading decision based on current market data
        """
        # Process market data
        input_tensor = self.process_market_data(market_data)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            try:
                # Reset model state
                if hasattr(self.model, 'reset_state'):
                    self.model.reset_state()
                
                # Forward pass
                output = self.model(input_tensor)
                
                # Handle different output formats
                if isinstance(output, tuple):
                    prediction = output[0]
                else:
                    prediction = output
                
                # Convert to probability-like values
                if prediction.dim() > 1:
                    prediction = prediction.mean(dim=1)  # Average across features
                
                prediction = torch.sigmoid(prediction)  # Convert to [0, 1]
                raw_signal = prediction.cpu().numpy().flatten()[0]
                
            except Exception as e:
                print(f"Model prediction error: {e}")
                raw_signal = 0.5  # Default to neutral
        
        # Calculate additional signals from technical indicators
        technical_signal = self._calculate_technical_signal(market_data)
        
        # Combine signals (70% model, 30% technical)
        combined_signal = 0.7 * raw_signal + 0.3 * technical_signal
        
        # Calculate confidence based on signal strength and consistency
        confidence = abs(combined_signal - 0.5) * 2  # Distance from neutral
        
        # Make trading decision
        if combined_signal > self.buy_threshold and confidence > self.confidence_threshold:
            # Buy signal
            position_size = min(0.3, confidence * 0.5)  # Size based on confidence
            action = TradingAction(
                action='buy',
                position_size=position_size,
                confidence=confidence,
                expected_return=combined_signal - 0.5
            )
        elif combined_signal < self.sell_threshold and confidence > self.confidence_threshold:
            # Sell signal
            position_size = min(1.0, confidence)  # Sell proportion based on confidence
            action = TradingAction(
                action='sell',
                position_size=position_size,
                confidence=confidence,
                expected_return=0.5 - combined_signal
            )
        else:
            # Hold
            action = TradingAction(
                action='hold',
                position_size=0.0,
                confidence=1.0 - confidence,  # High confidence in holding
                expected_return=0.0
            )
        
        return action
    
    def _calculate_technical_signal(self, market_data: pd.DataFrame) -> float:
        """Calculate technical analysis signal"""
        signals = []
        latest = market_data.iloc[-1]
        
        try:
            # RSI signal
            if 'RSI' in market_data.columns:
                rsi = latest['RSI']
                if rsi < 30:
                    signals.append(0.8)  # Oversold - buy signal
                elif rsi > 70:
                    signals.append(0.2)  # Overbought - sell signal
                else:
                    signals.append(0.5)  # Neutral
            
            # MACD signal
            if all(col in market_data.columns for col in ['MACD', 'MACD_Signal']):
                macd_diff = latest['MACD'] - latest['MACD_Signal']
                if macd_diff > 0:
                    signals.append(0.7)  # Bullish
                else:
                    signals.append(0.3)  # Bearish
            
            # Bollinger Bands signal
            if 'BB_Position' in market_data.columns:
                bb_pos = latest['BB_Position']
                if bb_pos < 0.2:
                    signals.append(0.8)  # Near lower band - buy
                elif bb_pos > 0.8:
                    signals.append(0.2)  # Near upper band - sell
                else:
                    signals.append(0.5)  # Neutral
            
            # Moving average signal
            if all(col in market_data.columns for col in ['Close', 'SMA_10', 'SMA_30']):
                price = latest['Close']
                sma_short = latest['SMA_10']
                sma_long = latest['SMA_30']
                
                if price > sma_short > sma_long:
                    signals.append(0.8)  # Strong uptrend
                elif price < sma_short < sma_long:
                    signals.append(0.2)  # Strong downtrend
                else:
                    signals.append(0.5)  # Mixed signals
            
        except Exception as e:
            print(f"Technical signal calculation error: {e}")
            signals = [0.5]  # Default to neutral
        
        return np.mean(signals) if signals else 0.5

class TradingSimulator:
    """
    Main trading simulation class that orchestrates the trading process
    """
    def __init__(self, model, device, config: Dict):
        self.model = model
        self.device = device
        self.config = config
        
        # Initialize components
        self.trading_env = TradingEnvironment(
            initial_capital=config.get('initial_capital', 100000),
            transaction_cost=config.get('transaction_cost', 0.001),
            slippage=config.get('slippage', 0.0005)
        )
        
        self.trading_agent = TradingAgent(
            model=model,
            device=device,
            confidence_threshold=config.get('confidence_threshold', 0.6)
        )
        
    def load_market_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load and prepare market data with technical indicators"""
        try:
            # Download data
            df = yf.download(ticker, start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Calculate technical indicators
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
            
            # Remove NaN values
            df_with_indicators = df_with_indicators.dropna()
            
            print(f"Loaded {len(df_with_indicators)} days of market data for {ticker}")
            print(f"Features: {list(df_with_indicators.columns)}")
            
            return df_with_indicators
            
        except Exception as e:
            print(f"Error loading market data: {e}")
            raise
    
    def run_simulation(self, ticker: str, start_date: str, end_date: str, 
                      sequence_length: int = 30) -> Dict:
        """
        Run the complete trading simulation
        """
        print(f"\n{'='*60}")
        print(f"TRADING SIMULATION: {ticker}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.trading_env.initial_capital:,.2f}")
        print(f"{'='*60}")
        
        # Load market data
        market_data = self.load_market_data(ticker, start_date, end_date)
        
        # Reset environment
        self.trading_env.reset()
        
        # Initialize tracking
        daily_decisions = []
        performance_metrics = []
        
        # Run simulation day by day
        for i in range(sequence_length, len(market_data)):
            # Get current date and price
            current_date = market_data.index[i]
            current_data = market_data.iloc[i-sequence_length:i]
            current_price = market_data.iloc[i]['Close']
            
            # Make trading decision
            try:
                trading_action = self.trading_agent.make_trading_decision(
                    current_data, 
                    self.trading_env.shares
                )
                
                # Execute trade
                trade_result = self.trading_env.execute_trade(
                    trading_action, 
                    current_price, 
                    current_date.strftime('%Y-%m-%d')
                )
                
                # Track decision
                daily_decisions.append({
                    'date': current_date,
                    'price': current_price,
                    'action': trading_action.action,
                    'position_size': trading_action.position_size,
                    'confidence': trading_action.confidence,
                    'portfolio_value': self.trading_env.get_portfolio_value(current_price),
                    'shares': self.trading_env.shares,
                    'cash': self.trading_env.cash
                })
                
                # Progress update
                if i % 50 == 0:
                    portfolio_value = self.trading_env.get_portfolio_value(current_price)
                    print(f"Day {i}: {current_date.strftime('%Y-%m-%d')} | "
                          f"Portfolio: ${portfolio_value:,.2f} | "
                          f"Action: {trading_action.action} | "
                          f"Price: ${current_price:.2f}")
                
            except Exception as e:
                print(f"Error on day {i}: {e}")
                continue
        
        # Calculate final performance
        final_portfolio_value = self.trading_env.get_portfolio_value(market_data.iloc[-1]['Close'])
        
        # Compile results
        results = self._compile_results(
            market_data, daily_decisions, final_portfolio_value
        )
        
        return results
    
    def _compile_results(self, market_data: pd.DataFrame, 
                        daily_decisions: List[Dict], 
                        final_portfolio_value: float) -> Dict:
        """Compile comprehensive trading results"""
        
        # Basic performance metrics
        initial_capital = self.trading_env.initial_capital
        total_return = (final_portfolio_value - initial_capital) / initial_capital
        total_fees = self.trading_env.total_fees
        
        # Calculate buy-and-hold benchmark
        start_price = market_data.iloc[0]['Close']
        end_price = market_data.iloc[-1]['Close']
        buy_hold_return = (end_price - start_price) / start_price
        buy_hold_final_value = initial_capital * (1 + buy_hold_return)
        
        # Trading activity analysis
        decisions_df = pd.DataFrame(daily_decisions)
        action_counts = decisions_df['action'].value_counts()
        
        # Risk metrics
        daily_returns = np.array(self.trading_env.daily_returns)
        volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (np.mean(daily_returns) * 252) / (volatility + 1e-8)
        
        # Maximum drawdown
        portfolio_values = np.array(self.trading_env.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        profitable_days = np.sum(daily_returns > 0)
        total_trading_days = len(daily_returns)
        win_rate = profitable_days / total_trading_days if total_trading_days > 0 else 0
        
        # Compile comprehensive results
        results = {
            # Basic Performance
            'initial_capital': initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return_pct': total_return * 100,
            'total_return_dollar': final_portfolio_value - initial_capital,
            'final_cash': self.trading_env.cash,
            'final_shares': self.trading_env.shares,
            'final_stock_value': self.trading_env.shares * end_price,
            
            # Benchmark Comparison
            'buy_hold_return_pct': buy_hold_return * 100,
            'buy_hold_final_value': buy_hold_final_value,
            'excess_return_pct': (total_return - buy_hold_return) * 100,
            'outperformed_market': total_return > buy_hold_return,
            
            # Trading Activity
            'total_trades': self.trading_env.total_trades,
            'total_fees': total_fees,
            'buy_actions': action_counts.get('buy', 0),
            'sell_actions': action_counts.get('sell', 0),
            'hold_actions': action_counts.get('hold', 0),
            'trading_days': len(daily_decisions),
            
            # Risk Metrics
            'volatility_annualized': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate * 100,
            
            # Detailed Data
            'daily_decisions': decisions_df,
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'market_data': market_data,
            'trades': self.trading_env.trades
        }
        
        return results
    
    def print_performance_summary(self, results: Dict):
        """Print a comprehensive performance summary"""
        print(f"\n{'='*80}")
        print(f"TRADING SIMULATION RESULTS")
        print(f"{'='*80}")
        
        # Performance Summary
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Initial Capital:        ${results['initial_capital']:>12,.2f}")
        print(f"   Final Portfolio Value:  ${results['final_portfolio_value']:>12,.2f}")
        print(f"   Total Return:           ${results['total_return_dollar']:>12,.2f} ({results['total_return_pct']:>+6.2f}%)")
        print(f"   Final Cash:             ${results['final_cash']:>12,.2f}")
        print(f"   Final Stock Value:      ${results['final_stock_value']:>12,.2f} ({results['final_shares']:>8.2f} shares)")
        
        # Benchmark Comparison
        print(f"\n📈 BENCHMARK COMPARISON:")
        print(f"   Buy & Hold Return:      ${results['buy_hold_final_value'] - results['initial_capital']:>12,.2f} ({results['buy_hold_return_pct']:>+6.2f}%)")
        print(f"   Strategy vs Market:     ${results['total_return_dollar'] - (results['buy_hold_final_value'] - results['initial_capital']):>12,.2f} ({results['excess_return_pct']:>+6.2f}%)")
        print(f"   Outperformed Market:    {'✅ YES' if results['outperformed_market'] else '❌ NO'}")
        
        # Trading Activity
        print(f"\n🔄 TRADING ACTIVITY:")
        print(f"   Total Trades:           {results['total_trades']:>12,}")
        print(f"   Trading Days:           {results['trading_days']:>12,}")
        print(f"   Buy Actions:            {results['buy_actions']:>12,}")
        print(f"   Sell Actions:           {results['sell_actions']:>12,}")
        print(f"   Hold Actions:           {results['hold_actions']:>12,}")
        print(f"   Total Fees:             ${results['total_fees']:>12,.2f}")
        
        # Risk Metrics
        print(f"\n⚠️  RISK METRICS:")
        print(f"   Annualized Volatility:  {results['volatility_annualized']:>12.2f}%")
        print(f"   Sharpe Ratio:           {results['sharpe_ratio']:>12.2f}")
        print(f"   Maximum Drawdown:       {results['max_drawdown_pct']:>12.2f}%")
        print(f"   Win Rate:               {results['win_rate_pct']:>12.2f}%")
        
        # Strategy Assessment
        print(f"\n🎯 STRATEGY ASSESSMENT:")
        
        # Performance rating
        if results['total_return_pct'] > 20:
            performance_rating = "🌟 EXCELLENT"
        elif results['total_return_pct'] > 10:
            performance_rating = "🟢 GOOD"
        elif results['total_return_pct'] > 0:
            performance_rating = "🟡 MODERATE"
        else:
            performance_rating = "🔴 POOR"
        
        print(f"   Performance Rating:     {performance_rating}")
        
        # Risk-adjusted performance
        if results['sharpe_ratio'] > 1.5:
            risk_rating = "🌟 EXCELLENT"
        elif results['sharpe_ratio'] > 1.0:
            risk_rating = "🟢 GOOD"
        elif results['sharpe_ratio'] > 0.5:
            risk_rating = "🟡 MODERATE"
        else:
            risk_rating = "🔴 POOR"
        
        print(f"   Risk-Adjusted Rating:   {risk_rating}")
        
        # Trading efficiency
        if results['total_trades'] > 0:
            avg_return_per_trade = results['total_return_dollar'] / results['total_trades']
            print(f"   Avg Return per Trade:   ${avg_return_per_trade:>12,.2f}")
        
        print(f"\n{'='*80}")
    
    def create_performance_visualizations(self, results: Dict, save_dir: str = "trading_results"):
        """Create comprehensive performance visualizations"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Portfolio Value Over Time
        ax1 = plt.subplot(3, 3, 1)
        dates = results['daily_decisions']['date']
        portfolio_values = results['portfolio_values']
        
        plt.plot(dates, portfolio_values, label='Strategy Portfolio', linewidth=2, color='blue')
        
        # Add buy/hold benchmark
        initial_capital = results['initial_capital']
        market_data = results['market_data']
        start_price = market_data.iloc[0]['Close']
        benchmark_values = [(price / start_price) * initial_capital for price in market_data['Close']]
        
        if len(dates) == len(benchmark_values):
            plt.plot(dates, benchmark_values, label='Buy & Hold', linewidth=2, color='red', alpha=0.7)
        
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 2. Daily Returns Distribution
        ax2 = plt.subplot(3, 3, 2)
        daily_returns = results['daily_returns'] * 100  # Convert to percentage
        plt.hist(daily_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(np.mean(daily_returns), color='red', linestyle='--', label=f'Mean: {np.mean(daily_returns):.2f}%')
        plt.title('Daily Returns Distribution')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 3. Drawdown Chart
        ax3 = plt.subplot(3, 3, 3)
        portfolio_values = np.array(results['portfolio_values'])
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max * 100
        
        plt.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        plt.plot(dates, drawdown, color='darkred', linewidth=1)
        plt.title('Portfolio Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.xticks(rotation=45)
        
        # 4. Trading Actions Over Time
        ax4 = plt.subplot(3, 3, 4)
        decisions_df = results['daily_decisions']
        
        # Create action timeline
        buy_dates = decisions_df[decisions_df['action'] == 'buy']['date']
        sell_dates = decisions_df[decisions_df['action'] == 'sell']['date']
        buy_prices = decisions_df[decisions_df['action'] == 'buy']['price']
        sell_prices = decisions_df[decisions_df['action'] == 'sell']['price']
        
        # Plot price with buy/sell markers
        plt.plot(dates, [results['market_data'].loc[date, 'Close'] for date in dates], 
                color='black', alpha=0.7, linewidth=1)
        
        if len(buy_dates) > 0:
            plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=50, label='Buy', alpha=0.8)
        if len(sell_dates) > 0:
            plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=50, label='Sell', alpha=0.8)
        
        plt.title('Trading Actions')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 5. Position Size Over Time
        ax5 = plt.subplot(3, 3, 5)
        position_values = [row['shares'] * row['price'] for _, row in decisions_df.iterrows()]
        position_percentages = [(pos_val / port_val) * 100 if port_val > 0 else 0 
                               for pos_val, port_val in zip(position_values, results['portfolio_values'])]
        
        plt.plot(dates, position_percentages, color='purple', linewidth=2)
        plt.title('Position Size Over Time')
        plt.ylabel('Position Size (% of Portfolio)')
        plt.xticks(rotation=45)
        
        # 6. Cumulative Returns Comparison
        ax6 = plt.subplot(3, 3, 6)
        strategy_returns = [(pv / results['initial_capital'] - 1) * 100 for pv in results['portfolio_values']]
        
        if len(dates) == len(benchmark_values):
            benchmark_returns = [(bv / results['initial_capital'] - 1) * 100 for bv in benchmark_values]
            plt.plot(dates, benchmark_returns, label='Buy & Hold', color='red', linewidth=2, alpha=0.7)
        
        plt.plot(dates, strategy_returns, label='Strategy', color='blue', linewidth=2)
        plt.title('Cumulative Returns Comparison')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 7. Monthly Returns Heatmap
        ax7 = plt.subplot(3, 3, 7)
        try:
            decisions_df['year_month'] = pd.to_datetime(decisions_df['date']).dt.to_period('M')
            monthly_returns = decisions_df.groupby('year_month')['portfolio_value'].apply(
                lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 else 0
            )
            
            # Create monthly returns matrix for heatmap
            monthly_data = monthly_returns.reset_index()
            monthly_data['year'] = monthly_data['year_month'].dt.year
            monthly_data['month'] = monthly_data['year_month'].dt.month
            
            pivot_data = monthly_data.pivot(index='year', columns='month', values='portfolio_value')
            
            if not pivot_data.empty:
                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax7)
                plt.title('Monthly Returns Heatmap (%)')
            else:
                plt.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', transform=ax7.transAxes)
                plt.title('Monthly Returns Heatmap')
        except Exception as e:
            plt.text(0.5, 0.5, f'Heatmap unavailable: {str(e)[:50]}...', ha='center', va='center', transform=ax7.transAxes)
            plt.title('Monthly Returns Heatmap')
        
        # 8. Risk-Return Scatter (if we have benchmark data)
        ax8 = plt.subplot(3, 3, 8)
        strategy_return = results['total_return_pct']
        strategy_vol = results['volatility_annualized'] * 100
        benchmark_return = results['buy_hold_return_pct']
        
        plt.scatter(strategy_vol, strategy_return, s=100, color='blue', label='Strategy', alpha=0.8)
        
        # Estimate benchmark volatility (simplified)
        if len(results['market_data']) > 1:
            market_returns = results['market_data']['Close'].pct_change().dropna()
            benchmark_vol = market_returns.std() * np.sqrt(252) * 100
            plt.scatter(benchmark_vol, benchmark_return, s=100, color='red', label='Buy & Hold', alpha=0.8)
        
        plt.xlabel('Volatility (%)')
        plt.ylabel('Return (%)')
        plt.title('Risk-Return Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Performance Metrics Summary (Text)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        metrics_text = f"""
Key Performance Metrics:

Total Return: {results['total_return_pct']:.2f}%
Benchmark Return: {results['buy_hold_return_pct']:.2f}%
Excess Return: {results['excess_return_pct']:.2f}%

Sharpe Ratio: {results['sharpe_ratio']:.2f}
Max Drawdown: {results['max_drawdown_pct']:.2f}%
Win Rate: {results['win_rate_pct']:.2f}%

Total Trades: {results['total_trades']:,}
Trading Fees: ${results['total_fees']:,.2f}

Final Portfolio: ${results['final_portfolio_value']:,.2f}
        """
        
        ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/trading_performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results to CSV
        decisions_df.to_csv(f"{save_dir}/daily_trading_decisions.csv", index=False)
        
        # Save trades to CSV
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df.to_csv(f"{save_dir}/executed_trades.csv", index=False)
        
        print(f"\n📁 Results saved to: {save_dir}/")
        print(f"   📊 Dashboard: trading_performance_dashboard.png")
        print(f"   📋 Daily decisions: daily_trading_decisions.csv")
        print(f"   💼 Executed trades: executed_trades.csv")

def run_comprehensive_trading_test(model, device, config=None):
    """
    Run a comprehensive trading simulation test
    """
    if config is None:
        config = {
            'initial_capital': 100000,
            'transaction_cost': 0.001,
            'slippage': 0.0005,
            'confidence_threshold': 0.6
        }
    
    # Initialize simulator
    simulator = TradingSimulator(model, device, config)
    
    # Test parameters
    test_configs = [
        {
            'ticker': 'AAPL',
            'start_date': '2022-01-01',
            'end_date': '2023-12-31',
            'name': 'Apple Inc. (2022-2023)'
        },
        {
            'ticker': 'MSFT',
            'start_date': '2022-01-01', 
            'end_date': '2023-12-31',
            'name': 'Microsoft Corp. (2022-2023)'
        },
        {
            'ticker': 'GOOGL',
            'start_date': '2022-01-01',
            'end_date': '2023-12-31', 
            'name': 'Alphabet Inc. (2022-2023)'
        }
    ]
    
    all_results = {}
    
    for test_config in test_configs:
        print(f"\n🚀 Starting simulation: {test_config['name']}")
        
        try:
            results = simulator.run_simulation(
                ticker=test_config['ticker'],
                start_date=test_config['start_date'],
                end_date=test_config['end_date']
            )
            
            # Print performance summary
            simulator.print_performance_summary(results)
            
            # Create visualizations
            save_dir = f"trading_results/{test_config['ticker']}_{test_config['start_date'][:4]}"
            simulator.create_performance_visualizations(results, save_dir)
            
            all_results[test_config['name']] = results
            
        except Exception as e:
            print(f"❌ Simulation failed for {test_config['name']}: {e}")
            continue
    
    # Create summary comparison
    if all_results:
        create_multi_stock_comparison(all_results)
    
    return all_results

def create_multi_stock_comparison(all_results: Dict):
    """Create a comparison chart across multiple stocks"""
    
    print(f"\n{'='*80}")
    print(f"MULTI-STOCK PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    comparison_data = []
    
    for name, results in all_results.items():
        comparison_data.append({
            'Stock': name,
            'Total Return (%)': results['total_return_pct'],
            'Buy & Hold (%)': results['buy_hold_return_pct'],
            'Excess Return (%)': results['excess_return_pct'],
            'Sharpe Ratio': results['sharpe_ratio'],
            'Max Drawdown (%)': results['max_drawdown_pct'],
            'Win Rate (%)': results['win_rate_pct'],
            'Total Trades': results['total_trades'],
            'Final Value ($)': results['final_portfolio_value']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("\n📊 PERFORMANCE COMPARISON TABLE:")
    print(comparison_df.round(2).to_string(index=False))
    
    # Calculate average performance
    avg_return = comparison_df['Total Return (%)'].mean()
    avg_sharpe = comparison_df['Sharpe Ratio'].mean()
    avg_excess = comparison_df['Excess Return (%)'].mean()
    outperformed_count = sum(comparison_df['Excess Return (%)'] > 0)
    total_tests = len(comparison_df)
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Average Return:         {avg_return:>8.2f}%")
    print(f"   Average Sharpe Ratio:   {avg_sharpe:>8.2f}")
    print(f"   Average Excess Return:  {avg_excess:>8.2f}%")
    print(f"   Market Outperformance:  {outperformed_count}/{total_tests} ({outperformed_count/total_tests*100:.1f}%)")
    
    # Overall assessment
    if avg_return > 15 and avg_sharpe > 1.0 and outperformed_count/total_tests >= 0.7:
        overall_rating = "🌟 EXCELLENT - Strong consistent performance"
    elif avg_return > 8 and avg_sharpe > 0.5 and outperformed_count/total_tests >= 0.5:
        overall_rating = "🟢 GOOD - Solid performance with room for improvement"
    elif avg_return > 0 and outperformed_count/total_tests >= 0.3:
        overall_rating = "🟡 MODERATE - Mixed results, needs optimization"
    else:
        overall_rating = "🔴 POOR - Significant improvement needed"
    
    print(f"\n🎯 OVERALL STRATEGY RATING: {overall_rating}")
    
    # Save comparison results
    comparison_df.to_csv("trading_results/multi_stock_comparison.csv", index=False)
    print(f"\n💾 Comparison saved to: trading_results/multi_stock_comparison.csv")
    
    return comparison_df

# Example usage and integration
if __name__ == "__main__":
    # Example of how to integrate with the existing brain-inspired neural network
    
    print("🧠 Brain-Inspired Neural Network Trading Simulation")
    print("=" * 60)
    
    # This would be called from your main training/testing pipeline
    # run_comprehensive_trading_test(model, device, config)