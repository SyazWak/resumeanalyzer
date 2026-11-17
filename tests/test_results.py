"""Tests for shared result assembly."""

from utils.ai_analyzer import (
    ATSOptimization,
    ComprehensiveAnalysis,
    MatchScore,
    SalaryEstimation,
    SkillGap,
)
from utils.results import build_results_dict, create_analysis_summary


def make_analysis(errors=None):
    return ComprehensiveAnalysis(
        match_score=MatchScore(78.5, 85, 75, 80, 70, 82, "good match"),
        skill_gap=SkillGap(["React"], {}, ["Learn React"], ["React"]),
        salary_estimation=SalaryEstimation(90000, 120000, 105000, [], "Upskill"),
        ats_optimization=ATSOptimization(82, 70, 80, ["Docker"], [], []),
        detailed_feedback={"detailed_feedback": "Looks good"},
        improvement_plan=["Add React"],
        next_steps=["Learn React"],
        errors=errors or [],
    )


def test_build_results_dict_shape():
    results = build_results_dict(make_analysis(), {"location": "Berlin"})

    assert results["metadata"]["ai_enhanced"] is True
    assert results["metadata"]["context"] == {"location": "Berlin"}
    assert "analysis_timestamp" in results["metadata"]
    assert results["ai_analysis"]["ai_match_score"].overall_score == 78.5
    assert results["summary"]["ai_overall_score"] == 78.5
    assert results["summary"]["recommendations"] == ["Add React"]
    assert results["summary"]["analysis_errors"] == []


def test_summary_includes_analysis_errors():
    results = build_results_dict(make_analysis(["Salary estimation: timeout"]))
    assert results["summary"]["analysis_errors"] == ["Salary estimation: timeout"]


def test_create_analysis_summary_without_match_score():
    summary = create_analysis_summary({"improvement_plan": ["Fix formatting"]})
    assert summary["recommendations"] == ["Fix formatting"]
    assert "ai_overall_score" not in summary
