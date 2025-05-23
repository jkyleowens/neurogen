#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Manager for Brain-Inspired Neural Network

This module provides intelligent configuration loading with environment-specific
overrides and validation.

Usage:
    from config_manager import ConfigManager
    
    config_manager = ConfigManager('config/comprehensive_config.yaml')
    config = config_manager.get_config(environment='development')
"""

import yaml
import os
import copy
from typing import Dict, Any, Optional

class ConfigManager:
    """Intelligent configuration manager with environment support."""
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the comprehensive configuration file
        """
        self.config_path = config_path
        self.base_config = None
        self.environments = {}
        self.load_config()
    
    def load_config(self):
        """Load the base configuration and environment overrides."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # Separate base config from environment overrides
            self.base_config = {k: v for k, v in full_config.items() if k != 'environments'}
            self.environments = full_config.get('environments', {})
            
            print(f"✅ Configuration loaded from {self.config_path}")
            print(f"📋 Available environments: {list(self.environments.keys())}")
            
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def get_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific environment.
        
        Args:
            environment: Environment name (development, testing, production)
                        If None, auto-detects based on various factors
        
        Returns:
            Complete configuration dictionary with environment overrides applied
        """
        if environment is None:
            environment = self.detect_environment()
        
        # Start with base configuration
        config = copy.deepcopy(self.base_config)
        
        # Apply environment-specific overrides
        if environment in self.environments:
            print(f"🔧 Applying {environment} environment overrides...")
            self._apply_overrides(config, self.environments[environment])
        else:
            print(f"⚠️  Environment '{environment}' not found, using base configuration")
        
        # Validate final configuration
        self._validate_config(config)
        
        return config
    
    def detect_environment(self) -> str:
        """
        Auto-detect the current environment based on various factors.
        
        Returns:
            Detected environment name
        """
        # Check environment variables
        env_var = os.environ.get('NEUROGEN_ENV', '').lower()
        if env_var in self.environments:
            return env_var
        
        # Check for development indicators
        if any(indicator in os.getcwd().lower() for indicator in ['dev', 'develop', 'test']):
            return 'development'
        
        # Check for testing indicators
        if any(indicator in os.getcwd().lower() for indicator in ['test', 'ci', 'github']):
            return 'testing'
        
        # Check for production indicators
        if any(indicator in os.environ for indicator in ['PRODUCTION', 'PROD', 'LIVE']):
            return 'production'
        
        # Default to development for safety
        print("🔍 Auto-detected environment: development (default)")
        return 'development'
    
    def _apply_overrides(self, base_config: Dict[str, Any], overrides: Dict[str, Any]):
        """Recursively apply environment overrides to base configuration."""
        for key, value in overrides.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                # Recursively apply nested overrides
                self._apply_overrides(base_config[key], value)
            else:
                # Direct override
                base_config[key] = value
                print(f"   Override: {key} = {value}")
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate the final configuration."""
        required_sections = ['model', 'training', 'trading']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section missing: {section}")
        
        # Validate specific requirements
        if config['model']['input_size'] <= 0:
            raise ValueError("Model input_size must be positive")
        
        if config['training']['num_epochs'] <= 0:
            raise ValueError("Training num_epochs must be positive")
        
        if config['trading']['initial_capital'] <= 0:
            raise ValueError("Trading initial_capital must be positive")
        
        print("✅ Configuration validation passed")
    
    def save_config(self, config: Dict[str, Any], output_path: str):
        """Save configuration to a file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            print(f"💾 Configuration saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
    
    def print_config_summary(self, config: Dict[str, Any]):
        """Print a summary of the current configuration."""
        print("\n📋 CONFIGURATION SUMMARY")
        print("=" * 50)
        
        # Model summary
        model = config.get('model', {})
        print(f"🧠 Model:")
        print(f"   Input Size: {model.get('input_size', 'N/A')}")
        print(f"   Hidden Size: {model.get('hidden_size', 'N/A')}")
        print(f"   Output Size: {model.get('output_size', 'N/A')}")
        print(f"   Use BioGRU: {model.get('use_bio_gru', 'N/A')}")
        
        # Training summary
        training = config.get('training', {})
        print(f"\n🎓 Training:")
        print(f"   Epochs: {training.get('num_epochs', 'N/A')}")
        print(f"   Batch Size: {training.get('batch_size', 'N/A')}")
        print(f"   Learning Rate: {training.get('learning_rate', 'N/A')}")
        print(f"   Learning Mode: {training.get('learning_mode', 'N/A')}")
        
        # Trading summary
        trading = config.get('trading', {})
        print(f"\n💰 Trading:")
        print(f"   Initial Capital: ${trading.get('initial_capital', 0):,}")
        print(f"   Transaction Cost: {trading.get('transaction_cost', 0)*100:.1f}%")
        print(f"   Max Position Size: {trading.get('max_position_size', 0)*100:.1f}%")
        print(f"   Confidence Threshold: {trading.get('confidence_threshold', 0)}")
        
        # Test scenarios summary
        scenarios = config.get('test_scenarios', [])
        print(f"\n📊 Test Scenarios: {len(scenarios)}")
        for i, scenario in enumerate(scenarios[:3], 1):  # Show first 3
            print(f"   {i}. {scenario.get('name', 'N/A')} ({scenario.get('ticker', 'N/A')})")
        if len(scenarios) > 3:
            print(f"   ... and {len(scenarios) - 3} more")
        
        print("=" * 50)

def create_environment_specific_configs():
    """Create separate config files for each environment."""
    config_manager = ConfigManager('config/comprehensive_config.yaml')
    
    environments = ['development', 'testing', 'production']
    
    for env in environments:
        print(f"\n📝 Creating {env} configuration...")
        config = config_manager.get_config(env)
        output_path = f"config/{env}_config.yaml"
        config_manager.save_config(config, output_path)

