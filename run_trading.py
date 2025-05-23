#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick Trading Simulation Runner

Simple script to run realistic trading simulations on the brain-inspired neural network.
This demonstrates the trading agent's performance with actual buy/sell/hold decisions.

Usage:
    python run_trading_test.py
    python run_trading_test.py --model-path path/to/model.pt
    python run_trading_test.py --quick-test
"""

import sys
import os
import torch
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import necessary modules
try:
    from src.model import BrainInspiredNN
    from trading_simulation import TradingSimulator, TechnicalIndicators
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required packages are installed:")
    print("pip install torch pandas numpy matplotlib seaborn yfinance scikit-learn")
    sys.exit(1)

def create_simple_model(device):
    """Create a simple model for demonstration if no trained model is available"""
    
    config = {
        'model': {
            'input_size': 23,  # Number of technical indicators
            'hidden_size': 64,
            'output_size': 1,
            'use_bio_gru': False  # Use simpler model for demo
        },
        'controller': {
            'num_layers': 2,
            'persistent_memory_size': 32,
            'dropout': 0.2
        },
        'neuromodulator': {
            'dopamine_scale': 1.0,
            'serotonin_scale': 1.0,
            'norepinephrine_scale': 1.0,
            'acetylcholine_scale': 1.0,
            'reward_decay': 0.95
        }
    }
    
    model = BrainInspiredNN(config).to(device)
    print("Created demo model (untrained - results will be random)")
    return model

def run_simple_trading_demo(model_path=None, quick_test=True):
    """Run a simple trading demonstration"""
    
    print("🧠 Brain-Inspired Neural Network Trading Demo")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create model
    if model_path and os.path.exists(model_path):
        try:
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Create model (simplified architecture detection)
            config = {
                'model': {
                    'input_size': 23,
                    'hidden_size': 128,
                    'output_size': 1,
                    'use_bio_gru': True
                }
            }
            
            model = BrainInspiredNN(config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.eval()
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Creating demo model instead...")
            model = create_simple_model(device)
    else:
        print("No model path provided or file not found. Creating demo model...")
        model = create_simple_model(device)
    
    # Trading configuration
    trading_config = {
        'initial_capital': 50000,  # $50K for demo
        'transaction_cost': 0.001,  # 0.1% per trade
        'slippage': 0.0005,        # 0.05% slippage
        'confidence_threshold': 0.5,  # Lower threshold for demo
        'max_position_size': 0.25   # Max 25% position
    }
    
    # Test scenarios
    if quick_test:
        test_scenarios = [
            {
                'name': 'AAPL_Recent',
                'ticker': 'AAPL',
                'start_date': '2023-06-01',
                'end_date': '2023-12-31',
                'description': 'Apple - Recent 6 months'
            }
        ]
    else:
        test_scenarios = [
            {
                'name': 'AAPL_2023',
                'ticker': 'AAPL',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'description': 'Apple - Full year 2023'
            },
            {
                'name': 'MSFT_2023',
                'ticker': 'MSFT', 
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'description': 'Microsoft - Full year 2023'
            }
        ]
    
    # Initialize trading simulator
    simulator = TradingSimulator(model, device, trading_config)
    
    all_results = {}
    
    # Run simulations
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"🚀 Running: {scenario['description']}")
        print(f"{'='*60}")
        
        try:
            # Run the simulation
            results = simulator.run_simulation(
                ticker=scenario['ticker'],
                start_date=scenario['start_date'], 
                end_date=scenario['end_date'],
                sequence_length=20  # Shorter sequence for demo
            )
            
            # Print results summary
            simulator.print_performance_summary(results)
            
            # Create visualizations
            save_dir = f"demo_results/{scenario['name']}"
            simulator.create_performance_visualizations(results, save_dir)
            
            all_results[scenario['name']] = results
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary across all tests
    if all_results:
        print(f"\n{'='*60}")
        print(f"📊 DEMO SUMMARY")
        print(f"{'='*60}")
        
        total_return_avg = np.mean([r['total_return_pct'] for r in all_results.values()])
        outperformed_count = sum([r['outperformed_market'] for r in all_results.values()])
        total_tests = len(all_results)
        
        print(f"Tests completed: {total_tests}")
        print(f"Average return: {total_return_avg:.2f}%")
        print(f"Market outperformance: {outperformed_count}/{total_tests} ({outperformed_count/total_tests*100:.1f}%)")
        
        # Best performing scenario
        best_scenario = max(all_results.items(), key=lambda x: x[1]['total_return_pct'])
        print(f"Best performance: {best_scenario[0]} ({best_scenario[1]['total_return_pct']:.2f}%)")
        
        print(f"\n📁 Results saved to demo_results/ directory")
        print(f"💡 To run with your own trained model: python run_trading_test.py --model-path your_model.pt")
        
    else:
        print(f"\n❌ No simulations completed successfully")
    
    return all_results

def detailed_performance_analysis(results):
    """Provide detailed analysis of trading performance"""
    
    if not results:
        return
    
    print(f"\n{'='*80}")
    print(f"📈 DETAILED PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    for scenario_name, result in results.items():
        print(f"\n🎯 {scenario_name}:")
        print(f"   📊 Financial Performance:")
        print(f"      • Start Capital:     ${result['initial_capital']:>10,.2f}")
        print(f"      • End Capital:       ${result['final_portfolio_value']:>10,.2f}")
        print(f"      • Total Gain/Loss:   ${result['total_return_dollar']:>10,.2f}")
        print(f"      • Return Rate:       {result['total_return_pct']:>10.2f}%")
        print(f"      • Buy & Hold Return: {result['buy_hold_return_pct']:>10.2f}%")
        print(f"      • Alpha Generated:   {result['excess_return_pct']:>10.2f}%")
        
        print(f"\n   📈 Trading Activity:")
        print(f"      • Total Trades:      {result['total_trades']:>10}")
        print(f"      • Buy Orders:        {result['buy_actions']:>10}")
        print(f"      • Sell Orders:       {result['sell_actions']:>10}")
        print(f"      • Hold Days:         {result['hold_actions']:>10}")
        print(f"      • Trading Fees:      ${result['total_fees']:>10.2f}")
        
        print(f"\n   ⚖️  Risk Metrics:")
        print(f"      • Volatility:        {result['volatility_annualized']*100:>10.2f}%")
        print(f"      • Sharpe Ratio:      {result['sharpe_ratio']:>10.2f}")
        print(f"      • Max Drawdown:      {result['max_drawdown_pct']:>10.2f}%")
        print(f"      • Win Rate:          {result['win_rate_pct']:>10.2f}%")
        
        # Performance assessment
        if result['total_return_pct'] > 15:
            perf_rating = "🌟 EXCELLENT"
        elif result['total_return_pct'] > 8:
            perf_rating = "🟢 GOOD"
        elif result['total_return_pct'] > 0:
            perf_rating = "🟡 MODERATE"
        else:
            perf_rating = "🔴 POOR"
        
        beat_market = "✅ YES" if result['outperformed_market'] else "❌ NO"
        
        print(f"\n   🏆 Assessment:")
        print(f"      • Performance Rating: {perf_rating}")
        print(f"      • Beat Market:        {beat_market}")
        
        # Trading insights
        if result['total_trades'] > 0:
            avg_return_per_trade = result['total_return_dollar'] / result['total_trades']
            print(f"      • Avg Return/Trade:   ${avg_return_per_trade:>10.2f}")
        
        # Final portfolio composition
        final_stock_value = result.get('final_stock_value', 0)
        final_cash = result.get('final_cash', 0)
        if final_stock_value + final_cash > 0:
            stock_allocation = (final_stock_value / (final_stock_value + final_cash)) * 100
            cash_allocation = (final_cash / (final_stock_value + final_cash)) * 100
            print(f"\n   💼 Final Portfolio:")
            print(f"      • Stock Position:     {stock_allocation:>10.1f}% (${final_stock_value:,.2f})")
            print(f"      • Cash Position:      {cash_allocation:>10.1f}% (${final_cash:,.2f})")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Trading Simulation Demo')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with single stock')
    parser.add_argument('--detailed-analysis', action='store_true', help='Show detailed performance analysis')
    
    args = parser.parse_args()
    
    try:
        # Run the trading demo
        results = run_simple_trading_demo(
            model_path=args.model_path,
            quick_test=args.quick_test
        )
        
        # Show detailed analysis if requested
        if args.detailed_analysis and results:
            detailed_performance_analysis(results)
        
        print(f"\n🎉 Demo completed successfully!")
        
        if not args.model_path:
            print(f"\n💡 NEXT STEPS:")
            print(f"   1. Train your brain-inspired neural network model")
            print(f"   2. Run: python run_trading_test.py --model-path your_trained_model.pt")
            print(f"   3. Use realistic_trading_test.py for comprehensive evaluation")
            print(f"   4. Optimize trading parameters based on results")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())