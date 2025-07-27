"""
Basic tests for the AI Resume Analyzer utilities.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.text_extractor import TextExtractor
from utils.text_processor import TextProcessor
from utils.similarity import SimilarityCalculator


def test_text_extractor_initialization():
    """Test TextExtractor can be initialized."""
    extractor = TextExtractor()
    assert extractor is not None
    assert len(extractor.get_supported_formats()) > 0


def test_text_processor_initialization():
    """Test TextProcessor can be initialized."""
    processor = TextProcessor()
    assert processor is not None


def test_similarity_calculator_initialization():
    """Test SimilarityCalculator can be initialized."""
    calculator = SimilarityCalculator()
    assert calculator is not None


def test_basic_similarity_calculation():
    """Test basic similarity calculation."""
    calculator = SimilarityCalculator()
    
    text1 = "python machine learning data science"
    text2 = "python programming artificial intelligence"
    
    similarity = calculator.calculate_cosine_similarity(text1, text2)
    assert 0 <= similarity <= 1


def test_keyword_extraction():
    """Test keyword extraction."""
    processor = TextProcessor()
    
    text = "Python developer with machine learning experience and Django skills"
    keywords = processor.extract_keywords(text, top_k=5)
    
    assert isinstance(keywords, list)
    assert len(keywords) <= 5


def test_text_cleaning():
    """Test text cleaning functionality."""
    processor = TextProcessor()
    
    dirty_text = "This is a TEST!!! with 123 numbers and @#$% symbols."
    clean_text = processor.clean_text(dirty_text)
    
    assert clean_text.islower()
    assert "123" not in clean_text


if __name__ == "__main__":
    pytest.main([__file__])
