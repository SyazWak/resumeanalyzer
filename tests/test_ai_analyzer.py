"""Tests for AdvancedAIAnalyzer module."""

from unittest.mock import MagicMock, patch

import pytest

from utils.ai_analyzer import (
    AdvancedAIAnalyzer,
    AnalysisConfig,
    MatchScore,
    coerce_score,
    coerce_string_list,
    parse_json_response,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    return AnalysisConfig(model="test-model", temperature=0.5, max_tokens=1000)


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
        mock_response.json.return_value = {"choices": [{"message": {"content": "test response"}}]}
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


class TestConfiguration:
    def test_config_reads_environment(self):
        env = {
            "AI_MODEL": "provider/model",
            "TEMPERATURE": "0.25",
            "MAX_TOKENS": "1234",
            "API_TIMEOUT": "5",
            "ENABLE_SALARY_ESTIMATION": "false",
            "DEFAULT_LOCATION": "Berlin",
            "DEFAULT_INDUSTRY": "Finance",
        }
        with patch.dict("os.environ", env):
            config = AnalysisConfig.from_env()
        assert config.model == "provider/model"
        assert config.temperature == 0.25
        assert config.max_tokens == 1234
        assert config.request_timeout == 5
        assert config.enable_salary_estimation is False
        assert config.default_location == "Berlin"
        assert config.default_industry == "Finance"

    def test_invalid_numbers_fall_back_to_defaults(self):
        with patch.dict("os.environ", {"TEMPERATURE": "not-a-number", "MAX_TOKENS": "oops"}):
            config = AnalysisConfig.from_env()
        assert config.temperature == 0.7
        assert config.max_tokens == 2000

    def test_explicit_config_wins_over_environment(self):
        config = AnalysisConfig(model="explicit-model")
        with patch.dict("os.environ", {"AI_MODEL": "env-model"}):
            analyzer = AdvancedAIAnalyzer(config)
        assert analyzer.config.model == "explicit-model"

    def test_placeholder_api_key_is_not_configured(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "your_openrouter_api_key_here"}):
            analyzer = AdvancedAIAnalyzer()
        assert analyzer.api_configured is False


class TestJsonParsingAndCoercion:
    @pytest.mark.parametrize(
        "response",
        [
            '{"overall_score": 85}',
            '```json\n{"overall_score": 85}\n```',
            'Here you go: {"overall_score": 85} thanks!',
            '{"overall_score": 85,}',
            "{'overall_score': 85}",
        ],
    )
    def test_parse_json_strategies(self, response):
        assert parse_json_response(response)["overall_score"] == 85

    def test_unparseable_response_is_flagged(self):
        assert parse_json_response("not json at all").get("_parse_failed") is True

    def test_score_coercion(self):
        assert coerce_score("85") == 85.0
        assert coerce_score(150) == 100.0
        assert coerce_score(None, 42.0) == 42.0

    def test_string_list_coercion(self):
        assert coerce_string_list("React") == ["React"]
        assert coerce_string_list([" AWS ", "", None]) == ["AWS"]
        assert coerce_string_list(123) == []

    @patch("utils.ai_analyzer.requests.post")
    def test_llm_scores_are_normalised(self, mock_post, analyzer):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"overall_score": "85", "technical_skills_score": 120}'}}
            ]
        }
        mock_post.return_value = mock_response

        score = analyzer._analyze_match_score("resume", "job")
        assert score.overall_score == 85.0
        assert score.technical_skills_score == 100.0


class TestErrorHandling:
    @patch.object(AdvancedAIAnalyzer, "_make_api_request", return_value="AUTH_ERROR: bad key")
    def test_auth_error_short_circuits_remaining_calls(self, mock_request, analyzer):
        analysis = analyzer.analyze_resume_comprehensive("resume", "job")
        assert analysis.errors
        assert all("Authentication failed" in error for error in analysis.errors)
        assert mock_request.call_count == 1
        assert analysis.match_score.overall_score == 50.0

    def test_single_failed_step_does_not_discard_other_results(self, analyzer):
        good_score = MatchScore(82, 80, 80, 80, 80, 80, "good")
        with patch.object(analyzer, "_analyze_match_score", return_value=good_score), patch.object(
            analyzer, "_analyze_skill_gap", side_effect=RuntimeError("boom")
        ), patch.object(analyzer, "_generate_detailed_feedback", return_value={}):
            analysis = analyzer.analyze_resume_comprehensive("resume", "job")

        assert analysis.match_score.overall_score == 82
        assert analysis.skill_gap.missing_skills == []
        assert any("Skill gap" in error for error in analysis.errors)

    def test_feature_flags_disable_steps(self):
        flags = {
            "OPENROUTER_API_KEY": "test-key",
            "ENABLE_DETAILED_ANALYSIS": "false",
            "ENABLE_IMPROVEMENT_SUGGESTIONS": "false",
        }
        with patch.dict("os.environ", flags):
            analyzer = AdvancedAIAnalyzer()

        with patch.object(
            analyzer,
            "_analyze_match_score",
            return_value=MatchScore(80, 80, 80, 80, 80, 80, "ok"),
        ), patch.object(analyzer, "_generate_detailed_feedback") as feedback:
            analysis = analyzer.analyze_resume_comprehensive("resume", "job")

        feedback.assert_not_called()
        assert analysis.improvement_plan == []
        assert analysis.detailed_feedback["detailed_feedback"] == "Detailed analysis is disabled."
