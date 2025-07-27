#!/usr/bin/env python3
"""
Test the DeepSeek model configuration
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Load environment variables
load_dotenv()

# Import our AI analyzer
from utils.ai_analyzer import AdvancedAIAnalyzer, AnalysisConfig

def test_model_configuration():
    """Test that the correct DeepSeek model is configured."""
    
    print("🧪 Testing DeepSeek Model Configuration")
    print("=" * 50)
    
    # Test environment variables
    model_from_env = os.getenv('DEEPSEEK_MODEL')
    api_key = os.getenv('OPENROUTER_API_KEY')
    base_url = os.getenv('OPENROUTER_BASE_URL')
    
    print(f"📋 Environment Variables:")
    print(f"   DEEPSEEK_MODEL: {model_from_env}")
    print(f"   OPENROUTER_API_KEY: {'✅ Set' if api_key else '❌ Not set'} (length: {len(api_key) if api_key else 0})")
    print(f"   OPENROUTER_BASE_URL: {base_url}")
    print()
    
    # Test default configuration
    config = AnalysisConfig()
    print(f"🔧 Default AnalysisConfig:")
    print(f"   Model: {config.model}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Max Tokens: {config.max_tokens}")
    print()
    
    # Test AI analyzer initialization
    try:
        analyzer = AdvancedAIAnalyzer()
        print(f"🤖 AdvancedAIAnalyzer Configuration:")
        print(f"   Model: {analyzer.config.model}")
        print(f"   API Key: {'✅ Loaded' if analyzer.api_key else '❌ Missing'}")
        print(f"   Base URL: {analyzer.base_url}")
        print()
        
        # Verify the exact model is being used
        expected_model = "deepseek/deepseek-chat-v3-0324:free"
        if analyzer.config.model == expected_model:
            print(f"✅ SUCCESS: Using correct model: {expected_model}")
        else:
            print(f"❌ ERROR: Expected {expected_model}, but got {analyzer.config.model}")
            
    except Exception as e:
        print(f"❌ ERROR initializing analyzer: {e}")
        return False
    
    print()
    print("🎯 Model Configuration Test Complete!")
    print("   Your AI Resume Analyzer is configured to use:")
    print(f"   📊 Model: {analyzer.config.model}")
    print("   💡 Note: Add credits at https://openrouter.ai/settings/credits to enable AI features")
    
    return True

if __name__ == "__main__":
    test_model_configuration()
