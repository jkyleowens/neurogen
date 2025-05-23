#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Realistic Trading Test Script for Brain-Inspired Neural Network

This script runs comprehensive trading simulations that evaluate the neural network
as an actual trading agent, making buy/sell/hold decisions with realistic constraints
and providing detailed financial performance analysis.
"""

import sys
import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the model and trading simulation
from src.model import BrainInspiredNN
from trading_system import TradingSimulator, run_comprehensive_trading_test

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return create_default_config()

def create_default_config():
    """Create default configuration for trading simulation"""
    return {
        'model': {
            'input_size': 64,
            'hidden_size': 128,
            'output_size': 1,
            'use_bio_gru': True
        },
        'trading': {
            'initial_capital': 100000,
            'transaction_cost': 0.001,  # 0.1% per trade
            'slippage': 0.0005,  # 0.05% slippage
            'confidence_threshold': 0.6,
            'max_position_size': 0.3,  # Max 30% of capital per position
            'min_trade_amount': 100.0
        },
        'test_scenarios': [
            {
                'name': 'Apple_Bull_Market',
                'ticker': 'AAPL',
                'start_date': '2020-01-01',
                'end_date': '2021-12-31',
                'description': 'Apple during COVID recovery bull market'
            },
            {
                'name': 'Apple_Bear_Market', 
                'ticker': 'AAPL',
                'start_date': '2022-01-01',
                'end_date': '2022-12-31',
                'description': 'Apple during 2022 bear market'
            },
            {
                'name': 'Microsoft_Mixed',
                'ticker': 'MSFT',
                'start_date': '2021-01-01',
                'end_date': '2023-12-31',
                'description': 'Microsoft through mixed market conditions'
            },
            {
                'name': 'Tesla_Volatile',
                'ticker': 'TSLA',
                'start_date': '2022-01-01',
                'end_date': '2023-12-31',
                'description': 'Tesla during high volatility period'
            }
        ]
    }

def load_trained_model(model_path, config, device):
    """Load a pre-trained model for trading simulation"""
    print(f"Loading model from {model_path}...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Detect model architecture from checkpoint
        state_dict = checkpoint['model_state_dict']
        detected_config = detect_model_architecture(state_dict, config)
        
        # Create model
        model = BrainInspiredNN(detected_config).to(device)
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Architecture: {detected_config['model']}")
        
        return model, detected_config
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Creating new model for testing...")
        
        # Create new model for testing
        model = BrainInspiredNN(config).to(device)
        return model, config

def detect_model_architecture(state_dict, fallback_config):
    """Detect model architecture from state dict"""
    config = fallback_config.copy()
    
    # Detect output layer dimensions
    if 'output_layer.weight' in state_dict:
        output_weight = state_dict['output_layer.weight']
        config['model']['output_size'] = output_weight.shape[0]
        config['model']['hidden_size'] = output_weight.shape[1]
    
    # Detect if using BioGRU
    has_bio_gru = any('output_neurons' in key for key in state_dict.keys())
    config['model']['use_bio_gru'] = has_bio_gru
    
    return config

def run_single_trading_test(model, device, test_config, trading_config):
    """Run a single trading simulation test"""
    
    print(f"\n🎯 Running Trading Test: {test_config['name']}")
    print(f"   Stock: {test_config['ticker']}")
    print(f"   Period: {test_config['start_date']} to {test_config['end_date']}")
    print(f"   Description: {test_config['description']}")
    
    # Initialize trading simulator
    simulator = TradingSimulator(model, device, trading_config)
    
    try:
        # Run simulation
        results = simulator.run_simulation(
            ticker=test_config['ticker'],
            start_date=test_config['start_date'],
            end_date=test_config['end_date'],
            sequence_length=30
        )
        
        # Print detailed results
        simulator.print_performance_summary(results)
        
        # Create visualizations
        save_dir = f"trading_results/{test_config['name']}"
        simulator.create_performance_visualizations(results, save_dir)
        
        return results
        
    except Exception as e:
        print(f"❌ Trading test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_comprehensive_trading_evaluation(model, device, config):
    """Run comprehensive trading evaluation across multiple scenarios"""
    
    print(f"\n{'='*80}")
    print(f"🏦 COMPREHENSIVE TRADING EVALUATION")
    print(f"{'='*80}")
    
    trading_config = config.get('trading', {})
    test_scenarios = config.get('test_scenarios', [])
    
    if not test_scenarios:
        print("❌ No test scenarios defined. Using default scenarios.")
        test_scenarios = create_default_config()['test_scenarios']
    
    # Run all trading tests
    all_results = {}
    successful_tests = 0
    total_tests = len(test_scenarios)
    
    for i, test_scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{total_tests}: {test_scenario['name']}")
        print(f"{'='*60}")
        
        results = run_single_trading_test(model, device, test_scenario, trading_config)
        
        if results is not None:
            all_results[test_scenario['name']] = results
            successful_tests += 1
        else:
            print(f"⚠️ Test {test_scenario['name']} failed")
    
    # Generate comprehensive comparison
    if successful_tests > 0:
        print(f"\n{'='*80}")
        print(f"📊 COMPREHENSIVE EVALUATION SUMMARY")
        print(f"Successful Tests: {successful_tests}/{total_tests}")
        print(f"{'='*80}")
        
        # Create detailed comparison
        comparison_results = create_detailed_comparison(all_results, trading_config)
        
        # Generate final assessment
        generate_final_assessment(comparison_results, trading_config)
        
    else:
        print(f"\n❌ All trading tests failed. Please check model and configuration.")
    
    return all_results

def create_detailed_comparison(all_results, trading_config):
    """Create detailed comparison across all trading tests"""
    
    # Compile comparison metrics
    comparison_data = []
    
    for test_name, results in all_results.items():
        comparison_data.append({
            'Test_Scenario': test_name,
            'Initial_Capital': results['initial_capital'],
            'Final_Portfolio_Value': results['final_portfolio_value'],
            'Total_Return_Pct': results['total_return_pct'],
            'Total_Return_Dollar': results['total_return_dollar'],
            'Buy_Hold_Return_Pct': results['buy_hold_return_pct'],
            'Excess_Return_Pct': results['excess_return_pct'],
            'Outperformed_Market': results['outperformed_market'],
            'Sharpe_Ratio': results['sharpe_ratio'],
            'Max_Drawdown_Pct': results['max_drawdown_pct'],
            'Win_Rate_Pct': results['win_rate_pct'],
            'Total_Trades': results['total_trades'],
            'Total_Fees': results['total_fees'],
            'Volatility_Pct': results['volatility_annualized'] * 100,
            'Final_Cash': results['final_cash'],
            'Final_Stock_Value': results['final_stock_value']
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print detailed comparison table
    print("\n📋 DETAILED PERFORMANCE COMPARISON:")
    print("=" * 120)
    
    # Format and display key metrics
    key_metrics = [
        'Test_Scenario', 'Total_Return_Pct', 'Buy_Hold_Return_Pct', 
        'Excess_Return_Pct', 'Sharpe_Ratio', 'Max_Drawdown_Pct', 
        'Win_Rate_Pct', 'Total_Trades'
    ]
    
    display_df = comparison_df[key_metrics].round(2)
    print(display_df.to_string(index=False, max_colwidth=20))
    
    # Calculate aggregate statistics
    print(f"\n📊 AGGREGATE STATISTICS:")
    print(f"   Average Return:           {comparison_df['Total_Return_Pct'].mean():>8.2f}%")
    print(f"   Average Excess Return:    {comparison_df['Excess_Return_Pct'].mean():>8.2f}%")
    print(f"   Average Sharpe Ratio:     {comparison_df['Sharpe_Ratio'].mean():>8.2f}")
    print(f"   Average Max Drawdown:     {comparison_df['Max_Drawdown_Pct'].mean():>8.2f}%")
    print(f"   Average Win Rate:         {comparison_df['Win_Rate_Pct'].mean():>8.2f}%")
    print(f"   Market Outperformance:    {comparison_df['Outperformed_Market'].sum()}/{len(comparison_df)} tests")
    
    # Best and worst performing scenarios
    best_return_idx = comparison_df['Total_Return_Pct'].idxmax()
    worst_return_idx = comparison_df['Total_Return_Pct'].idxmin()
    
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   Scenario: {comparison_df.iloc[best_return_idx]['Test_Scenario']}")
    print(f"   Return: {comparison_df.iloc[best_return_idx]['Total_Return_Pct']:.2f}%")
    print(f"   Final Value: ${comparison_df.iloc[best_return_idx]['Final_Portfolio_Value']:,.2f}")
    
    print(f"\n📉 WORST PERFORMANCE:")
    print(f"   Scenario: {comparison_df.iloc[worst_return_idx]['Test_Scenario']}")
    print(f"   Return: {comparison_df.iloc[worst_return_idx]['Total_Return_Pct']:.2f}%")
    print(f"   Final Value: ${comparison_df.iloc[worst_return_idx]['Final_Portfolio_Value']:,.2f}")
    
    # Save detailed comparison
    os.makedirs("trading_results", exist_ok=True)
    comparison_df.to_csv("trading_results/comprehensive_comparison.csv", index=False)
    print(f"\n💾 Detailed comparison saved to: trading_results/comprehensive_comparison.csv")
    
    return comparison_df

def generate_final_assessment(comparison_df, trading_config):
    """Generate final assessment and recommendations"""
    
    print(f"\n{'='*80}")
    print(f"🎯 FINAL TRADING ASSESSMENT")
    print(f"{'='*80}")
    
    # Calculate key metrics for assessment
    avg_return = comparison_df['Total_Return_Pct'].mean()
    avg_sharpe = comparison_df['Sharpe_Ratio'].mean()
    avg_excess = comparison_df['Excess_Return_Pct'].mean()
    win_rate = comparison_df['Win_Rate_Pct'].mean()
    market_outperformance_rate = comparison_df['Outperformed_Market'].mean()
    consistency = 1 - (comparison_df['Total_Return_Pct'].std() / abs(avg_return)) if avg_return != 0 else 0
    
    print(f"\n📈 PERFORMANCE SCORECARD:")
    
    # Return Performance
    if avg_return > 20:
        return_grade = "A+"
        return_assessment = "Exceptional returns"
    elif avg_return > 15:
        return_grade = "A"
        return_assessment = "Excellent returns"
    elif avg_return > 10:
        return_grade = "B+"
        return_assessment = "Good returns"
    elif avg_return > 5:
        return_grade = "B"
        return_assessment = "Moderate returns"
    elif avg_return > 0:
        return_grade = "C"
        return_assessment = "Modest returns"
    else:
        return_grade = "F"
        return_assessment = "Poor returns"
    
    print(f"   Return Performance:       {return_grade:>8} ({return_assessment})")
    
    # Risk-Adjusted Performance
    if avg_sharpe > 2.0:
        sharpe_grade = "A+"
        sharpe_assessment = "Outstanding risk-adjusted returns"
    elif avg_sharpe > 1.5:
        sharpe_grade = "A"
        sharpe_assessment = "Excellent risk management"
    elif avg_sharpe > 1.0:
        sharpe_grade = "B+"
        sharpe_assessment = "Good risk-adjusted performance"
    elif avg_sharpe > 0.5:
        sharpe_grade = "B"
        sharpe_assessment = "Moderate risk efficiency"
    elif avg_sharpe > 0:
        sharpe_grade = "C"
        sharpe_assessment = "Below average risk management"
    else:
        sharpe_grade = "F"
        sharpe_assessment = "Poor risk control"
    
    print(f"   Risk-Adjusted Performance: {sharpe_grade:>6} ({sharpe_assessment})")
    
    # Market Outperformance
    if market_outperformance_rate >= 0.8:
        market_grade = "A+"
        market_assessment = "Consistently beats market"
    elif market_outperformance_rate >= 0.6:
        market_grade = "A"
        market_assessment = "Usually beats market"
    elif market_outperformance_rate >= 0.4:
        market_grade = "B"
        market_assessment = "Sometimes beats market"
    elif market_outperformance_rate >= 0.2:
        market_grade = "C"
        market_assessment = "Rarely beats market"
    else:
        market_grade = "F"
        market_assessment = "Underperforms market"
    
    print(f"   Market Outperformance:    {market_grade:>8} ({market_assessment})")
    
    # Trading Efficiency
    avg_trades = comparison_df['Total_Trades'].mean()
    avg_fees = comparison_df['Total_Fees'].mean()
    fee_impact = (avg_fees / trading_config.get('initial_capital', 100000)) * 100
    
    if fee_impact < 0.5 and avg_trades > 10:
        trading_grade = "A"
        trading_assessment = "Efficient trading strategy"
    elif fee_impact < 1.0:
        trading_grade = "B"
        trading_assessment = "Reasonable trading activity"
    elif fee_impact < 2.0:
        trading_grade = "C"
        trading_assessment = "High trading costs"
    else:
        trading_grade = "D"
        trading_assessment = "Excessive trading"
    
    print(f"   Trading Efficiency:       {trading_grade:>8} ({trading_assessment})")
    
    # Overall Grade Calculation
    grades = {'A+': 4.3, 'A': 4.0, 'B+': 3.3, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0}
    overall_gpa = (grades[return_grade] + grades[sharpe_grade] + grades[market_grade] + grades[trading_grade]) / 4
    
    if overall_gpa >= 4.0:
        overall_grade = "A"
        overall_assessment = "EXCELLENT - Ready for live trading consideration"
    elif overall_gpa >= 3.5:
        overall_grade = "A-"
        overall_assessment = "VERY GOOD - Minor optimizations needed"
    elif overall_gpa >= 3.0:
        overall_grade = "B+"
        overall_assessment = "GOOD - Some improvements required"
    elif overall_gpa >= 2.5:
        overall_grade = "B"
        overall_assessment = "AVERAGE - Significant improvements needed"
    elif overall_gpa >= 2.0:
        overall_grade = "C"
        overall_assessment = "BELOW AVERAGE - Major revisions required"
    else:
        overall_grade = "F"
        overall_assessment = "POOR - Strategy needs complete overhaul"
    
    print(f"\n🏆 OVERALL STRATEGY GRADE: {overall_grade} ({overall_assessment})")
    
    # Detailed recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    recommendations = []
    
    if avg_return < 5:
        recommendations.append("• Improve signal generation - current returns are below market expectations")
    
    if avg_sharpe < 1.0:
        recommendations.append("• Enhance risk management - volatility is too high relative to returns")
    
    if market_outperformance_rate < 0.5:
        recommendations.append("• Refine market timing - strategy frequently underperforms buy-and-hold")
    
    if fee_impact > 1.5:
        recommendations.append("• Reduce trading frequency - transaction costs are eating into profits")
    
    if win_rate < 50:
        recommendations.append("• Improve trade selection - win rate suggests poor entry/exit timing")
    
    if consistency < 0.3:
        recommendations.append("• Increase strategy consistency - performance varies too much across scenarios")
    
    if avg_excess < 2:
        recommendations.append("• Boost alpha generation - strategy barely adds value over passive investing")
    
    if not recommendations:
        recommendations.append("• Strategy shows strong performance - consider live testing with small capital")
        recommendations.append("• Monitor performance in different market conditions")
        recommendations.append("• Consider portfolio diversification across multiple assets")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    # Risk warnings
    print(f"\n⚠️  RISK WARNINGS:")
    
    max_drawdown = comparison_df['Max_Drawdown_Pct'].max()
    if max_drawdown > 20:
        print(f"   • HIGH DRAWDOWN RISK: Maximum observed drawdown of {max_drawdown:.1f}%")
    
    if comparison_df['Total_Return_Pct'].std() > 15:
        print(f"   • HIGH VOLATILITY: Strategy returns vary significantly across scenarios")
    
    if any(comparison_df['Total_Return_Pct'] < -10):
        print(f"   • LOSS POTENTIAL: Strategy showed significant losses in some scenarios")
    
    print(f"   • PAST PERFORMANCE: Historical results do not guarantee future performance")
    print(f"   • MARKET CONDITIONS: Real trading may face different conditions than backtests")
    print(f"   • EXECUTION RISK: Actual trading may have higher costs and slippage")
    
    # Summary statistics for final report
    print(f"\n📋 FINAL SUMMARY:")
    print(f"   Tests Completed:          {len(comparison_df)}")
    print(f"   Average Capital Gain:     ${comparison_df['Total_Return_Dollar'].mean():,.2f}")
    print(f"   Best Single Performance:  {comparison_df['Total_Return_Pct'].max():.2f}%")
    print(f"   Worst Single Performance: {comparison_df['Total_Return_Pct'].min():.2f}%")
    print(f"   Success Rate:             {(comparison_df['Total_Return_Pct'] > 0).sum()}/{len(comparison_df)} positive returns")
    print(f"   Market Beat Rate:         {comparison_df['Outperformed_Market'].sum()}/{len(comparison_df)} outperformed")
    
    return {
        'overall_grade': overall_grade,
        'overall_gpa': overall_gpa,
        'return_grade': return_grade,
        'sharpe_grade': sharpe_grade,
        'market_grade': market_grade,
        'trading_grade': trading_grade,
        'recommendations': recommendations,
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'market_outperformance_rate': market_outperformance_rate
    }

def main():
    """Main function for realistic trading test"""
    parser = argparse.ArgumentParser(description='Realistic Trading Test for Brain-Inspired Neural Network')
    
    parser.add_argument('--model-path', type=str, 
                       default='neurogen/models/checkpoints/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, 
                       default='config/financial_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, 
                       default='trading_results',
                       help='Directory to save trading results')
    parser.add_argument('--initial-capital', type=float, 
                       default=100000,
                       help='Initial trading capital')
    parser.add_argument('--device', type=str, 
                       default='auto',
                       help='Device for model inference (cuda/cpu/auto)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run a quick test with reduced scenarios')
    
    args = parser.parse_args()
    
    print("🧠 Brain-Inspired Neural Network - Realistic Trading Test")
    print("=" * 70)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if 'trading' not in config:
        config['trading'] = {}
    config['trading']['initial_capital'] = args.initial_capital
    
    # Quick test mode
    if args.quick_test:
        print("🚀 Running quick test mode...")
        config['test_scenarios'] = [
            {
                'name': 'AAPL_Quick_Test',
                'ticker': 'AAPL',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'description': 'Apple quick test scenario'
            }
        ]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trained model
    model, model_config = load_trained_model(args.model_path, config, device)
    
    # Update config with detected model architecture
    config.update(model_config)
    
    # Run comprehensive trading evaluation
    all_results = run_comprehensive_trading_evaluation(model, device, config)
    
    if all_results:
        print(f"\n🎉 Trading evaluation completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}/")
        
        # Save configuration used
        with open(f"{args.output_dir}/test_config.yaml", 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    else:
        print(f"\n❌ Trading evaluation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())