"""
AI Resume Analyzer Utilities Package
=====================================

This package contains utility modules for:
- Text extraction from PDF, TXT, and DOCX files
- AI-powered analysis with DeepSeek
- Data visualization and reporting

Copyright (c) 2025 SyazWak
Licensed under the MIT License - see LICENSE file for details.

Modules:
--------
- text_extractor: Extract text from various file formats
- ai_analyzer: AI-powered analysis using DeepSeek model
- visualizer: Create charts and visual reports
"""

__version__ = "1.0.0"
__author__ = "AI Resume Analyzer"

# Import main utility classes for easy access
try:
    from .text_extractor import TextExtractor
    from .ai_analyzer import AdvancedAIAnalyzer
    from .visualizer import ReportVisualizer
    
    __all__ = [
        "TextExtractor",
        "AdvancedAIAnalyzer",
        "ReportVisualizer"
    ]
except ImportError as e:
    # Handle missing dependencies gracefully
    __all__ = []
    print(f"Warning: Some utilities may not be available due to missing dependencies: {e}")
