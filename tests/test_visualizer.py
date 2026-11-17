"""Tests for ReportVisualizer chart helpers."""

from utils.visualizer import ReportVisualizer


def test_matplotlib_import_is_independent_of_seaborn():
    # seaborn is optional; matplotlib availability must not depend on it.
    from utils.visualizer import MATPLOTLIB_AVAILABLE

    assert isinstance(MATPLOTLIB_AVAILABLE, bool)


def test_score_gauge_uses_percentage_scale():
    viz = ReportVisualizer()
    fig = viz.create_score_gauge(82.5, "Match")
    assert fig is not None
    assert fig.data[0].value == 82.5


def test_radar_chart_has_five_components():
    viz = ReportVisualizer()
    fig = viz.create_match_radar(
        {
            "technical_skills_score": 80,
            "experience_score": 70,
            "education_score": 90,
            "soft_skills_score": 60,
            "ats_score": 75,
        }
    )
    assert fig is not None
    assert len(fig.data[0].theta) == 5
    assert list(fig.data[0].r) == [80, 70, 90, 60, 75]


def test_breakdown_chart_normalises_scores():
    viz = ReportVisualizer()
    fig = viz.create_ai_score_breakdown({"technical_skills_score": "85", "ats_score": 120})
    assert fig is not None
    assert list(fig.data[0].y) == [85, 0, 0, 0, 100]
