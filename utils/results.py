"""Shared assembly of analysis results for both the CLI and the web UI."""

from datetime import datetime
from typing import Any, Dict, Optional

from utils.ai_analyzer import ComprehensiveAnalysis


def create_analysis_summary(ai_results: Dict[str, Any]) -> Dict[str, Any]:
    """Build the summary block from an ``ai_analysis`` result dictionary."""
    summary: Dict[str, Any] = {
        "recommendations": list(ai_results.get("improvement_plan") or []),
        "analysis_errors": list(ai_results.get("errors") or []),
    }

    match_score = ai_results.get("ai_match_score")
    if match_score is not None:
        summary["ai_overall_score"] = match_score.overall_score
        summary["ai_breakdown"] = {
            "technical_skills": match_score.technical_skills_score,
            "experience": match_score.experience_score,
            "education": match_score.education_score,
            "soft_skills": match_score.soft_skills_score,
            "ats_compatibility": match_score.ats_score,
        }

    return summary


def build_results_dict(
    analysis: ComprehensiveAnalysis,
    context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a ``ComprehensiveAnalysis`` into the JSON-serialisable result shape."""
    ai_results: Dict[str, Any] = {
        "ai_match_score": analysis.match_score,
        "skill_gap_analysis": analysis.skill_gap,
        "salary_estimation": analysis.salary_estimation,
        "ats_optimization": analysis.ats_optimization,
        "ai_feedback": analysis.detailed_feedback,
        "improvement_plan": analysis.improvement_plan,
        "next_steps": analysis.next_steps,
        "errors": list(analysis.errors),
    }

    result_metadata: Dict[str, Any] = {
        "analysis_timestamp": datetime.now().isoformat(),
        "ai_enhanced": True,
        "context": context or {},
    }
    if metadata:
        result_metadata.update(metadata)

    return {
        "metadata": result_metadata,
        "ai_analysis": ai_results,
        "summary": create_analysis_summary(ai_results),
    }
