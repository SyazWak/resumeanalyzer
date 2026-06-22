"""Tests for AdvancedAIAnalyzer module."""

import json
import pytest
from unittest.mock import patch, MagicMock
from utils.ai_analyzer import AdvancedAIAnalyzer, AnalysisConfig


@pytest.fixture
def config():
    """Create a test configuration."""
    return AnalysisConfig(
        model="test-model",
        temperature=0.5,
        max_tokens=1000
    )


@pytest.fixture
def analyzer(config):
    """Create an AdvancedAIAnalyzer with mocked API key."""
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
        return AdvancedAIAnalyzer(config)


class TestAnalyzerInit:
    def test_api_key_loaded(self, analyzer):
        assert analyzer.api_key == "test-key-12345"

    def test_model_from_config(self, analyzer):
        assert analyzer.config.model == "test-model"

    def test_fallback_models_empty(self, analyzer):
        assert analyzer.fallback_models == []


class TestMakeApiRequest:
    def test_no_api_key(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}):
            analyzer = AdvancedAIAnalyzer()
            result = analyzer._make_api_request("test prompt")
            assert "unavailable" in result.lower()

    @patch("utils.ai_analyzer.requests.post")
    def test_successful_request(self, mock_post, analyzer):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        mock_post.return_value = mock_response

        result = analyzer._make_api_request("test prompt")
        assert result == "test response"

    @patch("utils.ai_analyzer.requests.post")
    def test_auth_error(self, mock_post, analyzer):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        result = analyzer._make_api_request("test prompt")
        assert "AUTH_ERROR" in result

    @patch("utils.ai_analyzer.requests.post")
    def test_rate_limit_with_fallback(self, mock_post, analyzer):
        analyzer.fallback_models = ["fallback-model"]

        # First call (primary) returns 429
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.text = "Rate limited"

        # Second call (fallback) returns 200
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "choices": [{"message": {"content": "fallback response"}}]
        }

        mock_post.side_effect = [rate_limit_response, success_response]

        result = analyzer._make_api_request("test prompt")
        assert result == "fallback response"


class TestModelStatus:
    def test_get_model_status(self, analyzer):
        status = analyzer.get_model_status()
        assert status["primary_model"] == "test-model"
        assert status["api_configured"] is True
        assert status["total_models_available"] == 1
