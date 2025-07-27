"""
Report Visualizer Module
=======================

Creates visualizations and reports for resume analysis results.
Supports both static plots and interactive Streamlit components.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import json

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
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
    
    def __init__(self, style: str = 'seaborn', color_palette: str = 'viridis'):
        """
        Initialize the ReportVisualizer.
        
        Args:
            style (str): Matplotlib style to use
            color_palette (str): Color palette for visualizations
        """
        self.style = style
        self.color_palette = color_palette
        
        # Set up matplotlib if available
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')  # Use default since seaborn might not be available
            if hasattr(sns, 'set_palette'):
                sns.set_palette(color_palette)
    
    def create_score_gauge(self, score: float, title: str = "Match Score") -> Optional[Any]:
        """
        Create a gauge chart for the match score.
        
        Args:
            score (float): Score between 0 and 1
            title (str): Chart title
            
        Returns:
            Optional[Any]: Plotly figure object or None if not available
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        percentage = score * 100
        
        # Determine color based on score
        if percentage >= 80:
            color = "green"
        elif percentage >= 60:
            color = "yellow"
        elif percentage >= 40:
            color = "orange"
        else:
            color = "red"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 70},  # Reference line at 70%
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=400, width=400)
        return fig
    
    def create_keyword_comparison(self, 
                                common_keywords: List[str], 
                                missing_keywords: List[str],
                                max_display: int = 10) -> Optional[Any]:
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
            rows=1, cols=2,
            subplot_titles=('Matching Keywords', 'Missing Keywords'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Common keywords
        if common_display:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(common_display))),
                    y=[1] * len(common_display),  # All have same height
                    text=common_display,
                    textposition='inside',
                    name="Found",
                    marker_color='green',
                    hovertext=common_display
                ),
                row=1, col=1
            )
        
        # Missing keywords
        if missing_display:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(missing_display))),
                    y=[1] * len(missing_display),  # All have same height
                    text=missing_display,
                    textposition='inside',
                    name="Missing",
                    marker_color='red',
                    hovertext=missing_display
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title="Keyword Analysis",
            showlegend=False,
            height=400
        )
        
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
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                text=counts,
                textposition='auto',
                marker_color='skyblue'
            )
        ])
        
        fig.update_layout(
            title="Skills by Category",
            xaxis_title="Skill Categories",
            yaxis_title="Number of Skills",
            height=400
        )
        
        return fig
    
    def create_score_breakdown(self, score_data: Dict[str, float]) -> Optional[Any]:
        """
        Create a breakdown of how the final score was calculated.
        
        Args:
            score_data (Dict[str, float]): Score components
            
        Returns:
            Optional[Any]: Plotly figure object or None if not available
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        # Extract score components
        cosine = score_data.get('cosine_contribution', 0) * 100
        keyword = score_data.get('keyword_contribution', 0) * 100
        jaccard = score_data.get('jaccard_contribution', 0) * 100
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Cosine Similarity (40%)', 'Keyword Match (40%)', 'Jaccard Similarity (20%)'],
                y=[cosine, keyword, jaccard],
                text=[f'{cosine:.1f}%', f'{keyword:.1f}%', f'{jaccard:.1f}%'],
                textposition='auto',
                marker_color=['lightblue', 'lightgreen', 'lightcoral']
            )
        ])
        
        fig.update_layout(
            title="Score Breakdown",
            xaxis_title="Components",
            yaxis_title="Contribution (%)",
            height=400
        )
        
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
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            return fig
        except Exception as e:
            logger.warning(f"Failed to create word cloud: {e}")
            return None
    
    def generate_text_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive text report.
        
        Args:
            analysis_results (Dict[str, Any]): Complete analysis results
            
        Returns:
            str: Formatted text report
        """
        final_score = analysis_results.get('final_percentage', 0)
        keyword_analysis = analysis_results.get('keyword_analysis', {})
        
        report = f"""
🎯 RESUME ANALYSIS REPORT
========================

📊 OVERALL MATCH SCORE: {final_score:.1f}%

🔍 DETAILED BREAKDOWN:
- Cosine Similarity: {analysis_results.get('cosine_similarity', 0)*100:.1f}%
- Keyword Match: {keyword_analysis.get('keyword_match_score', 0)*100:.1f}%
- Jaccard Similarity: {analysis_results.get('jaccard_similarity', 0)*100:.1f}%

✅ MATCHING KEYWORDS ({keyword_analysis.get('common_count', 0)} found):
"""
        
        # Add common keywords
        common_keywords = keyword_analysis.get('common_keywords', [])
        if common_keywords:
            for keyword in common_keywords[:10]:  # Show top 10
                report += f"   • {keyword}\n"
        else:
            report += "   • No matching keywords found\n"
        
        report += f"\n❌ MISSING KEYWORDS ({keyword_analysis.get('missing_count', 0)} total):\n"
        
        # Add missing keywords
        missing_keywords = keyword_analysis.get('missing_keywords', [])
        if missing_keywords:
            for keyword in missing_keywords[:10]:  # Show top 10
                report += f"   • {keyword}\n"
        else:
            report += "   • No missing keywords identified\n"
        
        # Add insights
        if 'insights' in analysis_results:
            report += "\n💡 INSIGHTS & RECOMMENDATIONS:\n"
            for insight in analysis_results['insights']:
                report += f"   {insight}\n"
        
        report += f"\n📈 KEYWORD COVERAGE: {keyword_analysis.get('coverage_percentage', 0):.1f}%"
        
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
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
            return False
    
    def display_streamlit_report(self, analysis_results: Dict[str, Any]) -> None:
        """
        Display interactive report in Streamlit.
        
        Args:
            analysis_results (Dict[str, Any]): Analysis results
        """
        if not STREAMLIT_AVAILABLE:
            logger.warning("Streamlit not available for interactive report")
            return
        
        st.title("🎯 Resume Analysis Report")
        
        # Main score display
        final_score = analysis_results.get('final_percentage', 0)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Match", f"{final_score:.1f}%")
        
        with col2:
            keyword_analysis = analysis_results.get('keyword_analysis', {})
            st.metric("Keywords Found", keyword_analysis.get('common_count', 0))
        
        with col3:
            st.metric("Keywords Missing", keyword_analysis.get('missing_count', 0))
        
        # Score gauge
        gauge_fig = self.create_score_gauge(final_score / 100)
        if gauge_fig:
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Keyword analysis
        st.subheader("🔍 Keyword Analysis")
        keyword_fig = self.create_keyword_comparison(
            keyword_analysis.get('common_keywords', []),
            keyword_analysis.get('missing_keywords', [])
        )
        if keyword_fig:
            st.plotly_chart(keyword_fig, use_container_width=True)
        
        # Score breakdown
        st.subheader("📊 Score Breakdown")
        breakdown_fig = self.create_score_breakdown(
            analysis_results.get('score_breakdown', {})
        )
        if breakdown_fig:
            st.plotly_chart(breakdown_fig, use_container_width=True)
        
        # Text report
        st.subheader("📝 Detailed Report")
        text_report = self.generate_text_report(analysis_results)
        st.text(text_report)
    
    def get_available_features(self) -> Dict[str, bool]:
        """Get information about available visualization features."""
        return {
            'matplotlib': MATPLOTLIB_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE,
            'streamlit': STREAMLIT_AVAILABLE,
            'wordcloud': WORDCLOUD_AVAILABLE
        }
