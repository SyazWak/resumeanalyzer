"""
Report Visualizer Module
=======================

Creates visualizations and reports for resume analysis results.
Supports both static plots and interactive Streamlit components.

Copyright (c) 2025 SyazWak
Licensed under the MIT License - see LICENSE file for details.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from utils.ai_analyzer import coerce_score

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportVisualizer:
    """
    Create visualizations and reports for resume analysis results.

    Supports multiple visualization libraries and output formats.
    """

    def __init__(
        self, style: str = "seaborn", color_palette: str = "viridis", theme: str = "light"
    ):
        """
        Initialize the ReportVisualizer.

        Args:
            style (str): Matplotlib style to use
            color_palette (str): Color palette for visualizations
            theme (str): Theme ('light' or 'dark') for chart backgrounds
        """
        self.style = style
        self.color_palette = color_palette
        self.theme = theme

        # Set up matplotlib if available (no seaborn dependency)
        if MATPLOTLIB_AVAILABLE:
            plt.style.use("default")

    def streamlit_layout(
        self, fig: Any, title: str = "", height: int = 400, y_range: Optional[List[float]] = None
    ) -> Any:
        """Make a Plotly figure readable in the active light/dark Streamlit theme."""
        theme = self.get_theme_config()
        font_color = theme["font_color"]
        yaxis: Dict[str, Any] = {"gridcolor": theme["grid_color"], "color": font_color}
        if y_range:
            yaxis["range"] = y_range

        fig.update_layout(
            title=title,
            height=height,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=font_color),
            xaxis=dict(gridcolor=theme["grid_color"], color=font_color),
            yaxis=yaxis,
            legend=dict(font=dict(color=font_color)),
        )
        return fig

    def get_theme_config(self) -> Dict[str, Any]:
        """
        Get theme configuration for charts.

        Returns:
            Dict[str, Any]: Theme configuration with colors and styling
        """
        if self.theme == "dark":
            return {
                "bg_color": "#0e1117",
                "paper_bgcolor": "#262730",
                "plot_bgcolor": "#262730",
                "text_color": "#ffffff",
                "grid_color": "#464853",
                "font_color": "#ffffff",
            }
        else:
            return {
                "bg_color": "#ffffff",
                "paper_bgcolor": "#ffffff",
                "plot_bgcolor": "#ffffff",
                "text_color": "#000000",
                "grid_color": "#dee2e6",
                "font_color": "#000000",
            }

    def apply_theme_layout(self, fig: Any) -> Any:
        """
        Apply theme styling to a plotly figure.

        Args:
            fig: Plotly figure object

        Returns:
            Any: Updated figure with theme styling
        """
        theme_config = self.get_theme_config()

        fig.update_layout(
            paper_bgcolor=theme_config["paper_bgcolor"],
            plot_bgcolor=theme_config["plot_bgcolor"],
            font=dict(color=theme_config["font_color"]),
            xaxis=dict(gridcolor=theme_config["grid_color"], color=theme_config["font_color"]),
            yaxis=dict(gridcolor=theme_config["grid_color"], color=theme_config["font_color"]),
            title=dict(font=dict(color=theme_config["font_color"])),
        )

        return fig

    def create_score_gauge(self, score: float, title: str = "Match Score") -> Optional[Any]:
        """
        Create a gauge chart for a score expressed as a percentage (0-100).

        Returns None when Plotly is not available.
        """
        if not PLOTLY_AVAILABLE:
            return None

        value = max(0.0, min(100.0, float(score)))
        color = (
            "green"
            if value >= 80
            else "yellow"
            if value >= 60
            else "orange"
            if value >= 40
            else "red"
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": title},
                delta={"reference": 70},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "lightgray"},
                        {"range": [40, 70], "color": "gray"},
                        {"range": [70, 100], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )

        fig.update_layout(height=400, width=400)
        return self.streamlit_layout(fig)

    def create_match_radar(
        self, score_data: Dict[str, float], title: str = "Score Breakdown"
    ) -> Optional[Any]:
        """Create a radar chart of the five analysis score components."""
        if not PLOTLY_AVAILABLE:
            return None

        categories = ["Technical Skills", "Experience", "Education", "Soft Skills", "ATS Score"]
        keys = [
            "technical_skills_score",
            "experience_score",
            "education_score",
            "soft_skills_score",
            "ats_score",
        ]
        values = [coerce_score(score_data.get(key)) for key in keys]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself", name="Your Score"))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
        )
        return self.streamlit_layout(fig, title=title, height=400)

    def create_keyword_comparison(
        self, common_keywords: List[str], missing_keywords: List[str], max_display: int = 10
    ) -> Optional[Any]:
        """
        Create a comparison chart of keywords.

        Args:
            common_keywords (List[str]): Keywords found in both resume and job
            missing_keywords (List[str]): Keywords missing from resume
            max_display (int): Maximum number of keywords to display

        Returns:
            Optional[Any]: Plotly figure object or None if not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Limit keywords for better visualization
        common_display = common_keywords[:max_display]
        missing_display = missing_keywords[:max_display]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Matching Keywords", "Missing Keywords"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        # Common keywords
        if common_display:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(common_display))),
                    y=[1] * len(common_display),  # All have same height
                    text=common_display,
                    textposition="inside",
                    name="Found",
                    marker_color="green",
                    hovertext=common_display,
                ),
                row=1,
                col=1,
            )

        # Missing keywords
        if missing_display:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(missing_display))),
                    y=[1] * len(missing_display),  # All have same height
                    text=missing_display,
                    textposition="inside",
                    name="Missing",
                    marker_color="red",
                    hovertext=missing_display,
                ),
                row=1,
                col=2,
            )

        fig.update_layout(title="Keyword Analysis", showlegend=False, height=400)

        # Hide y-axis since it's not meaningful
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_xaxes(showticklabels=False)

        return fig

    def create_skills_breakdown(self, skills_data: Dict[str, List[str]]) -> Optional[Any]:
        """
        Create a breakdown of skills by category.

        Args:
            skills_data (Dict[str, List[str]]): Skills grouped by category

        Returns:
            Optional[Any]: Plotly figure object or None if not available
        """
        if not PLOTLY_AVAILABLE or not skills_data:
            return None

        categories = list(skills_data.keys())
        counts = [len(skills) for skills in skills_data.values()]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=categories, y=counts, text=counts, textposition="auto", marker_color="skyblue"
                )
            ]
        )

        fig.update_layout(
            title="Skills by Category",
            xaxis_title="Skill Categories",
            yaxis_title="Number of Skills",
            height=400,
        )

        return fig

    def create_ai_score_breakdown(self, score_data: Dict[str, float]) -> Optional[Any]:
        """
        Create a breakdown of AI analysis components.

        Args:
            score_data (Dict[str, float]): AI score components

        Returns:
            Optional[Any]: Plotly figure object or None if not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Extract and normalise AI score components
        technical = coerce_score(score_data.get("technical_skills_score"))
        experience = coerce_score(score_data.get("experience_score"))
        education = coerce_score(score_data.get("education_score"))
        soft_skills = coerce_score(score_data.get("soft_skills_score"))
        ats_score = coerce_score(score_data.get("ats_score"))

        fig = go.Figure(
            data=[
                go.Bar(
                    x=[
                        "Technical Skills",
                        "Experience",
                        "Education",
                        "Soft Skills",
                        "ATS Optimization",
                    ],
                    y=[technical, experience, education, soft_skills, ats_score],
                    text=[
                        f"{technical:.1f}%",
                        f"{experience:.1f}%",
                        f"{education:.1f}%",
                        f"{soft_skills:.1f}%",
                        f"{ats_score:.1f}%",
                    ],
                    textposition="auto",
                    marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
                )
            ]
        )

        fig = self.streamlit_layout(fig, title="AI Analysis Score Breakdown", y_range=[0, 100])
        fig.update_layout(xaxis_title="Analysis Components", yaxis_title="Score (%)")
        return fig

    def create_wordcloud(self, text: str, title: str = "Word Cloud") -> Optional[Any]:
        """
        Create a word cloud from text.

        Args:
            text (str): Text to create word cloud from
            title (str): Title for the word cloud

        Returns:
            Optional[Any]: Matplotlib figure or None if not available
        """
        if not WORDCLOUD_AVAILABLE or not MATPLOTLIB_AVAILABLE or not text.strip():
            return None

        try:
            wordcloud = WordCloud(
                width=800, height=400, background_color="white", max_words=100, colormap="viridis"
            ).generate(text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=16, fontweight="bold")

            return fig
        except Exception as e:
            logger.warning(f"Failed to create word cloud: {e}")
            return None

    def generate_ai_text_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive AI analysis text report.

        Args:
            analysis_results (Dict[str, Any]): Complete AI analysis results

        Returns:
            str: Formatted text report
        """
        ai_analysis = analysis_results.get("ai_analysis", {})
        match_score = ai_analysis.get("ai_match_score", {})
        skill_gap = ai_analysis.get("skill_gap_analysis", {})
        salary_est = ai_analysis.get("salary_estimation", {})

        overall_score = match_score.get("overall_score", 0)

        report = f"""
🤖 AI RESUME ANALYSIS REPORT
===========================

📊 OVERALL AI MATCH SCORE: {overall_score:.1f}%

🔍 DETAILED AI BREAKDOWN:
- Technical Skills: {match_score.get("technical_skills_score", 0):.1f}%
- Experience Level: {match_score.get("experience_score", 0):.1f}%
- Education Match: {match_score.get("education_score", 0):.1f}%
- Soft Skills: {match_score.get("soft_skills_score", 0):.1f}%
- ATS Optimization: {match_score.get("ats_score", 0):.1f}%

❌ SKILL GAP ANALYSIS:
"""

        # Add missing skills
        missing_skills = skill_gap.get("missing_skills", [])
        if missing_skills:
            report += f"Missing Skills ({len(missing_skills)} identified):\n"
            for skill in missing_skills[:10]:
                report += f"   • {skill}\n"
        else:
            report += "   • No critical skills missing\n"

        # Add priority skills
        priority_skills = skill_gap.get("priority_skills", [])
        if priority_skills:
            report += "\n🎯 PRIORITY SKILLS TO DEVELOP:\n"
            for skill in priority_skills[:5]:
                report += f"   • {skill}\n"

        # Add salary information
        if salary_est:
            min_sal = salary_est.get("estimated_range_min", 0)
            max_sal = salary_est.get("estimated_range_max", 0)
            avg_sal = salary_est.get("market_average", 0)

            if min_sal and max_sal:
                report += f"\n💰 ESTIMATED SALARY RANGE: ${min_sal:,} - ${max_sal:,}"
                if avg_sal:
                    report += f"\n📈 MARKET AVERAGE: ${avg_sal:,}"

        return report

    def save_report_json(self, analysis_results: Dict[str, Any], filepath: str) -> bool:
        """
        Save analysis results as JSON file.

        Args:
            analysis_results (Dict[str, Any]): Analysis results
            filepath (str): Output file path

        Returns:
            bool: Success status
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
            return False

    def display_ai_streamlit_report(self, analysis_results: Dict[str, Any]) -> None:
        """
        Display interactive AI analysis report in Streamlit.

        Args:
            analysis_results (Dict[str, Any]): AI analysis results
        """
        if not STREAMLIT_AVAILABLE:
            logger.warning("Streamlit not available for interactive report")
            return

        st.title("🤖 AI Resume Analysis Report")

        # Extract AI analysis data
        ai_analysis = analysis_results.get("ai_analysis", {})
        match_score = ai_analysis.get("ai_match_score", {})
        skill_gap = ai_analysis.get("skill_gap_analysis", {})

        # Main score display
        overall_score = match_score.get("overall_score", 0)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🎯 AI Match Score", f"{overall_score:.1f}%")

        with col2:
            missing_skills = skill_gap.get("missing_skills", [])
            st.metric("❌ Missing Skills", len(missing_skills))

        with col3:
            ats_score = match_score.get("ats_score", 0)
            st.metric("🔍 ATS Score", f"{ats_score:.1f}%")

        # AI Score gauge
        gauge_fig = self.create_score_gauge(overall_score, "AI Match Score")
        if gauge_fig:
            st.plotly_chart(gauge_fig, use_container_width=True)

        # AI Score breakdown
        st.subheader("� AI Analysis Breakdown")
        breakdown_fig = self.create_ai_score_breakdown(match_score)
        if breakdown_fig:
            st.plotly_chart(breakdown_fig, use_container_width=True)

        # Skill gap analysis
        if missing_skills:
            st.subheader("🎯 Skill Gap Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Missing Skills:**")
                for skill in missing_skills[:10]:
                    st.markdown(f"• {skill}")

            with col2:
                priority_skills = skill_gap.get("priority_skills", [])
                if priority_skills:
                    st.markdown("**Priority Skills:**")
                    for skill in priority_skills[:5]:
                        st.markdown(f"• {skill}")

        # AI Text report
        st.subheader("📝 Detailed AI Analysis")
        ai_report = self.generate_ai_text_report(analysis_results)
        st.text(ai_report)

    def get_available_features(self) -> Dict[str, bool]:
        """Get information about available visualization features."""
        return {
            "matplotlib": MATPLOTLIB_AVAILABLE,
            "plotly": PLOTLY_AVAILABLE,
            "streamlit": STREAMLIT_AVAILABLE,
            "wordcloud": WORDCLOUD_AVAILABLE,
        }
