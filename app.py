"""
AI-Enhanced Resume Analyzer - Streamlit Web Application
=======================================================

Advanced web interface for intelligent resume analysis with AI-powered insights.

Copyright (c) 2025 SyazWak
Licensed under the MIT License - see LICENSE file for details.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.text_extractor import TextExtractor
from utils.visualizer import ReportVisualizer
from utils.ai_analyzer import AdvancedAIAnalyzer


# Configure Streamlit page
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-detect Streamlit theme
@st.cache_data
def get_system_theme():
    """Get system theme preference with multiple detection methods."""
    
    # Method 1: Check URL parameters for explicit theme setting
    try:
        query_params = st.query_params
        if 'theme' in query_params:
            theme = str(query_params['theme']).lower()
            if theme in ['dark', 'light']:
                return theme
    except:
        pass
    
    # Method 2: Try to detect from browser/system (limited in Streamlit)
    # This is a placeholder for more advanced detection
    
    # Method 3: Default to light mode
    # The JavaScript detection will help users see the right theme,
    # and they can always override in the sidebar
    return 'light'

def sync_with_streamlit_theme():
    """Sync our theme with user preferences and system detection."""
    
    # Priority 1: Manual user override in sidebar
    if 'manual_theme' in st.session_state:
        if st.session_state.theme != st.session_state.manual_theme:
            st.session_state.theme = st.session_state.manual_theme
        return
    
    # Priority 2: URL parameter override
    try:
        query_params = st.query_params
        if 'theme' in query_params:
            url_theme = str(query_params['theme']).lower()
            if url_theme in ['dark', 'light'] and st.session_state.theme != url_theme:
                st.session_state.theme = url_theme
                return
    except:
        pass
    
    # Priority 3: Auto-detection (system preference)
    detected_theme = get_system_theme()
    if st.session_state.get('theme') != detected_theme:
        st.session_state.theme = detected_theme

# Initialize theme
if 'theme' not in st.session_state:
    st.session_state.theme = get_system_theme()

# Sync theme preferences
sync_with_streamlit_theme()

# Theme configurations
themes = {
    'light': {
        'bg_color': '#ffffff',
        'text_color': '#000000',
        'secondary_bg': '#f8f9fa',
        'border_color': '#dee2e6',
        'accent_color': '#2196F3',
        'success_color': '#28a745',
        'warning_color': '#ffc107',
        'danger_color': '#dc3545',
        'ai_insight_bg': '#f0f8ff',
        'skill_gap_bg': '#fff3cd',
        'recommendation_bg': '#d4edda',
        'card_bg': '#ffffff',
        'shadow': 'rgba(0,0,0,0.1)'
    },
    'dark': {
        'bg_color': '#0e1117',
        'text_color': '#ffffff',
        'secondary_bg': '#262730',
        'border_color': '#464853',
        'accent_color': '#4dabf7',
        'success_color': '#51cf66',
        'warning_color': '#ffd43b',
        'danger_color': '#ff6b6b',
        'ai_insight_bg': '#1a2332',
        'skill_gap_bg': '#2d2a1f',
        'recommendation_bg': '#1f2d1f',
        'card_bg': '#262730',
        'shadow': 'rgba(0,0,0,0.3)'
    }
}

current_theme = themes[st.session_state.theme]

# Custom CSS with theme support
st.markdown(f"""
<style>
    /* Global theme variables */
    :root {{
        --bg-color: {current_theme['bg_color']};
        --text-color: {current_theme['text_color']};
        --secondary-bg: {current_theme['secondary_bg']};
        --border-color: {current_theme['border_color']};
        --accent-color: {current_theme['accent_color']};
        --success-color: {current_theme['success_color']};
        --warning-color: {current_theme['warning_color']};
        --danger-color: {current_theme['danger_color']};
        --card-bg: {current_theme['card_bg']};
        --shadow: {current_theme['shadow']};
    }}
    
    /* Auto-detect Streamlit theme */
    <script>
    function updateThemeFromStreamlit() {{
        // Check Streamlit's current theme by examining the body or main app background
        const stApp = document.querySelector('.stApp');
        if (stApp) {{
            const styles = window.getComputedStyle(stApp);
            const bgColor = styles.backgroundColor;
            
            // Parse RGB values
            const rgb = bgColor.match(/\\d+/g);
            if (rgb) {{
                const [r, g, b] = rgb.map(Number);
                const brightness = (r * 299 + g * 587 + b * 114) / 1000;
                
                // Determine if we're in dark mode
                const isDark = brightness < 128;
                const currentTheme = isDark ? 'dark' : 'light';
                
                // Store in sessionStorage for Python to read
                sessionStorage.setItem('streamlit_detected_theme', currentTheme);
                
                // Also set a data attribute for CSS targeting
                document.documentElement.setAttribute('data-streamlit-theme', currentTheme);
            }}
        }}
        
        // Also check system preference as fallback
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (!sessionStorage.getItem('streamlit_detected_theme')) {{
            sessionStorage.setItem('streamlit_detected_theme', systemPrefersDark ? 'dark' : 'light');
            document.documentElement.setAttribute('data-streamlit-theme', systemPrefersDark ? 'dark' : 'light');
        }}
    }}
    
    // Run detection when DOM is ready
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', updateThemeFromStreamlit);
    }} else {{
        updateThemeFromStreamlit();
    }}
    
    // Also run periodically to catch theme changes
    setInterval(updateThemeFromStreamlit, 1000);
    </script>
    
    /* Main app styling */
    .stApp {{
        background-color: {current_theme['bg_color']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {current_theme['secondary_bg']} !important;
    }}
    
    /* Main content area */
    .main > div {{
        padding-top: 2rem;
        background-color: {current_theme['bg_color']} !important;
    }}
    
    /* Metric containers */
    .metric-container {{
        background: linear-gradient(90deg, {current_theme['success_color']}, {current_theme['accent_color']});
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px {current_theme['shadow']};
    }}
    
    /* AI insight boxes */
    .ai-insight {{
        background: {current_theme['ai_insight_bg']} !important;
        border-left: 4px solid {current_theme['accent_color']};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: {current_theme['text_color']} !important;
        box-shadow: 0 2px 4px {current_theme['shadow']};
    }}
    
    /* Skill gap boxes */
    .skill-gap {{
        background: {current_theme['skill_gap_bg']} !important;
        border-left: 4px solid {current_theme['warning_color']};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: {current_theme['text_color']} !important;
        box-shadow: 0 2px 4px {current_theme['shadow']};
    }}
    
    /* Recommendation boxes */
    .recommendation {{
        background: {current_theme['recommendation_bg']} !important;
        border-left: 4px solid {current_theme['success_color']};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: {current_theme['text_color']} !important;
        box-shadow: 0 2px 4px {current_theme['shadow']};
    }}
    
    /* Cards and containers */
    .stContainer, .stColumn {{
        background-color: {current_theme['bg_color']} !important;
    }}
    
    /* Text elements */
    .stMarkdown, .stText, .stWrite {{
        color: {current_theme['text_color']} !important;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {current_theme['text_color']} !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {current_theme['accent_color']} !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        background-color: {current_theme['success_color']} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px {current_theme['shadow']} !important;
    }}
    
    /* File uploader */
    .stFileUploader {{
        background-color: {current_theme['card_bg']} !important;
        border: 2px dashed {current_theme['border_color']} !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }}
    
    /* Text areas */
    .stTextArea textarea {{
        background-color: {current_theme['card_bg']} !important;
        color: {current_theme['text_color']} !important;
        border: 1px solid {current_theme['border_color']} !important;
        border-radius: 5px !important;
    }}
    
    /* Select boxes */
    .stSelectbox select {{
        background-color: {current_theme['card_bg']} !important;
        color: {current_theme['text_color']} !important;
        border: 1px solid {current_theme['border_color']} !important;
    }}
    
    /* Alerts */
    .stAlert {{
        background-color: {current_theme['card_bg']} !important;
        color: {current_theme['text_color']} !important;
        border: 1px solid {current_theme['border_color']} !important;
        border-radius: 5px !important;
    }}
    
    /* Success messages */
    .stSuccess {{
        background-color: {current_theme['success_color']}20 !important;
        border-left: 4px solid {current_theme['success_color']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    /* Error messages */
    .stError {{
        background-color: {current_theme['danger_color']}20 !important;
        border-left: 4px solid {current_theme['danger_color']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    /* Warning messages */
    .stWarning {{
        background-color: {current_theme['warning_color']}20 !important;
        border-left: 4px solid {current_theme['warning_color']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    /* Info messages */
    .stInfo {{
        background-color: {current_theme['accent_color']}20 !important;
        border-left: 4px solid {current_theme['accent_color']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    /* Theme toggle button */
    .theme-toggle {{
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
        background: {current_theme['accent_color']} !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        font-size: 20px !important;
        cursor: pointer !important;
        box-shadow: 0 2px 10px {current_theme['shadow']} !important;
        transition: all 0.3s ease !important;
    }}
    
    .theme-toggle:hover {{
        background: {current_theme['success_color']} !important;
        transform: scale(1.1) !important;
    }}
    
    /* Plotly charts theme adjustment */
    .js-plotly-plot {{
        background-color: {current_theme['bg_color']} !important;
    }}
    
    /* Code blocks */
    .stCode {{
        background-color: {current_theme['secondary_bg']} !important;
        color: {current_theme['text_color']} !important;
        border: 1px solid {current_theme['border_color']} !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {current_theme['card_bg']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: {current_theme['secondary_bg']} !important;
        color: {current_theme['text_color']} !important;
    }}
</style>
""", unsafe_allow_html=True)


class EnhancedStreamlitApp:
    """Enhanced Streamlit application with AI capabilities."""
    
    def __init__(self):
        """Initialize the application components."""
        self.text_extractor = TextExtractor()
        self.visualizer = ReportVisualizer(theme=st.session_state.theme)
        
        # Initialize AI analyzer
        self.ai_analyzer = None
        self.ai_available = False
        
        try:
            self.ai_analyzer = AdvancedAIAnalyzer()
            self.ai_available = True
        except Exception as e:
            st.sidebar.warning(f"AI Analysis unavailable: {str(e)}")
    
    def run(self):
        """Main application runner."""
        # Header
        st.title("🤖 AI-Enhanced Resume Analyzer")
        st.markdown("*Intelligent resume analysis with advanced AI insights*")
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Analysis", 
            "🎯 AI Insights", 
            "📊 Visualizations", 
            "💾 Results"
        ])
        
        with tab1:
            self.render_analysis_tab()
        
        with tab2:
            self.render_ai_insights_tab()
        
        with tab3:
            self.render_visualizations_tab()
        
        with tab4:
            self.render_results_tab()
    
    def render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.title("⚙️ AI Configuration")
        
        # Theme settings
        st.sidebar.subheader("🎨 Theme Settings")
        
        # Show current theme status
        current_theme_display = "🌙 Dark Mode" if st.session_state.theme == 'dark' else "☀️ Light Mode"
        st.sidebar.info(f"**Current Theme:** {current_theme_display}")
        
        # Theme selection
        theme_options = {
            "Auto (System Preference)": "auto",
            "☀️ Light Mode": "light", 
            "🌙 Dark Mode": "dark"
        }
        
        # Determine current selection
        if 'manual_theme' not in st.session_state:
            current_selection = "Auto (System Preference)"
        else:
            manual = st.session_state.manual_theme
            current_selection = "☀️ Light Mode" if manual == 'light' else "🌙 Dark Mode"
        
        selected_theme = st.sidebar.selectbox(
            "Theme Preference:",
            options=list(theme_options.keys()),
            index=list(theme_options.keys()).index(current_selection),
            help="Choose your preferred theme. Auto mode follows your system/browser preference."
        )
        
        # Apply theme selection
        selected_value = theme_options[selected_theme]
        
        if selected_value == "auto":
            # Remove manual override
            if 'manual_theme' in st.session_state:
                del st.session_state.manual_theme
                st.rerun()
        else:
            # Set manual theme
            if st.session_state.get('manual_theme') != selected_value:
                st.session_state.manual_theme = selected_value
                st.session_state.theme = selected_value
                st.rerun()
        
        # Add helpful info
        if selected_theme == "Auto (System Preference)":
            st.sidebar.caption("💡 Theme automatically matches your system/browser dark mode setting")
        
        st.sidebar.divider()
        
        # AI settings
        st.sidebar.subheader("🤖 AI Model Configuration")
        if self.ai_available:
            st.sidebar.success("✅ AI Analysis Ready")
            
            # Show primary model
            primary_model = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat-v3-0324:free')
            st.sidebar.info(f"**Primary:** {primary_model}")
            
            # Show fallback models
            fallback_models = os.getenv('FALLBACK_MODELS', '')
            if fallback_models:
                fallback_list = [model.strip() for model in fallback_models.split(',')]
                st.sidebar.caption("**Fallback Models:**")
                for i, model in enumerate(fallback_list[:3], 1):  # Show first 3
                    st.sidebar.caption(f"  {i}. {model}")
                if len(fallback_list) > 3:
                    st.sidebar.caption(f"  +{len(fallback_list) - 3} more...")
            
            st.sidebar.caption("💡 Automatically switches to fallback models when rate limited")
        else:
            st.sidebar.error("❌ AI Analysis Unavailable")
            st.sidebar.warning("Please configure your OpenRouter API key")
        
        # Analysis context
        st.sidebar.subheader("Analysis Context")
        st.session_state.location = st.sidebar.text_input(
            "Location (for salary estimation)", 
            placeholder="e.g., San Francisco, CA"
        )
        st.session_state.industry = st.sidebar.selectbox(
            "Industry",
            ["", "Technology", "Finance", "Healthcare", "Marketing", "Engineering", "Other"]
        )
        
        # Analysis settings
        st.sidebar.subheader("Analysis Settings")
        st.session_state.analysis_depth = st.sidebar.selectbox(
            "Analysis Depth",
            ["Standard", "Comprehensive"],
            index=0
        )
        
        # Clear results button
        if st.sidebar.button("🗑️ Clear Results"):
            self.clear_session_state()
            st.rerun()
    
    def render_analysis_tab(self):
        """Render the main analysis tab."""
        st.header("📄 Document Analysis")
        
        # Create two columns for separate input choices
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Resume Input")
            resume_input_method = st.radio(
                "Choose resume input method:",
                ["📁 Upload File", "✏️ Text Input"],
                key="resume_input_method",
                horizontal=True
            )
            
            if resume_input_method == "📁 Upload File":
                self.render_resume_file_upload()
            else:
                self.render_resume_text_input()
        
        with col2:
            st.subheader("📋 Job Description Input")
            job_input_method = st.radio(
                "Choose job description input method:",
                ["📁 Upload File", "✏️ Text Input"],
                key="job_input_method",
                horizontal=True
            )
            
            if job_input_method == "📁 Upload File":
                self.render_job_file_upload()
            else:
                self.render_job_text_input()
        
        # Analysis button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Analyze Resume", use_container_width=True, type="primary"):
                self.perform_analysis()
    
    def render_resume_file_upload(self):
        """Render resume file upload interface."""
        st.info("📁 **Supported formats:** PDF, TXT, DOCX")
        resume_file = st.file_uploader(
            "Upload resume file",
            type=['pdf', 'txt', 'docx'],
            key="resume_file",
            help="Supported formats: PDF, TXT, DOCX"
        )
        if resume_file:
            file_type = resume_file.type if hasattr(resume_file, 'type') else 'Unknown'
            st.write(f"📄 **File:** {resume_file.name} ({file_type})")
            with st.spinner("Extracting text from resume..."):
                st.session_state.resume_text = self.extract_text_from_upload(resume_file)
                if st.session_state.resume_text:
                    st.success("✅ Resume text extracted successfully!")
                    word_count = len(st.session_state.resume_text.split())
                    st.write(f"📊 **Stats:** {len(st.session_state.resume_text)} characters, ~{word_count} words")
                    with st.expander("📄 Preview Resume Text"):
                        st.text_area("Resume Content", st.session_state.resume_text[:500] + "...", height=150, disabled=True)
                else:
                    st.error("❌ Failed to extract text from resume file")
    
    def render_resume_text_input(self):
        """Render resume text input interface."""
        st.info("✏️ **Direct input:** Copy and paste your resume text")
        resume_text = st.text_area(
            "Paste or type your resume text:",
            height=300,
            key="resume_text_input",
            placeholder="Paste your resume content here..."
        )
        if resume_text.strip():
            st.session_state.resume_text = resume_text
            word_count = len(resume_text.split())
            st.success(f"✅ Resume text ready ({len(resume_text)} characters, ~{word_count} words)")
    
    def render_job_file_upload(self):
        """Render job description file upload interface."""
        st.info("📁 **Supported formats:** PDF, TXT, DOCX")
        job_file = st.file_uploader(
            "Upload job description file",
            type=['pdf', 'txt', 'docx'],
            key="job_file",
            help="Supported formats: PDF, TXT, DOCX"
        )
        if job_file:
            file_type = job_file.type if hasattr(job_file, 'type') else 'Unknown'
            st.write(f"📋 **File:** {job_file.name} ({file_type})")
            with st.spinner("Extracting text from job description..."):
                st.session_state.job_text = self.extract_text_from_upload(job_file)
                if st.session_state.job_text:
                    st.success("✅ Job description text extracted successfully!")
                    word_count = len(st.session_state.job_text.split())
                    st.write(f"📊 **Stats:** {len(st.session_state.job_text)} characters, ~{word_count} words")
                    with st.expander("📋 Preview Job Description"):
                        st.text_area("Job Content", st.session_state.job_text[:500] + "...", height=150, disabled=True)
                else:
                    st.error("❌ Failed to extract text from job description file")
    
    def render_job_text_input(self):
        """Render job description text input interface."""
        st.info("✏️ **Direct input:** Copy and paste the job description")
        job_text = st.text_area(
            "Paste or type the job description:",
            height=300,
            key="job_text_input",
            placeholder="Paste the job description here..."
        )
        if job_text.strip():
            st.session_state.job_text = job_text
            word_count = len(job_text.split())
            st.success(f"✅ Job description ready ({len(job_text)} characters, ~{word_count} words)")
    
    def render_ai_insights_tab(self):
        """Render AI insights tab."""
        st.header("🎯 AI-Powered Insights")
        
        if not hasattr(st.session_state, 'analysis_results') or not st.session_state.analysis_results:
            st.info("🔍 Run an analysis first to see AI insights here!")
            return
        
        ai_analysis = st.session_state.analysis_results.get('ai_analysis')
        if not ai_analysis:
            st.warning("⚠️ No AI analysis available. Ensure AI is enabled and try again.")
            return
        
        # AI Match Score
        if 'ai_match_score' in ai_analysis:
            self.render_ai_match_score(ai_analysis['ai_match_score'])
        
        # Skill Gap Analysis
        if 'skill_gap_analysis' in ai_analysis:
            self.render_skill_gap_analysis(ai_analysis['skill_gap_analysis'])
        
        # Salary Estimation
        if 'salary_estimation' in ai_analysis and ai_analysis['salary_estimation']:
            self.render_salary_estimation(ai_analysis['salary_estimation'])
        
        # ATS Optimization
        if 'ats_optimization' in ai_analysis:
            self.render_ats_optimization(ai_analysis['ats_optimization'])
        
        # AI Feedback and Recommendations
        if 'ai_feedback' in ai_analysis:
            self.render_ai_feedback(ai_analysis)
    
    def render_ai_match_score(self, match_score):
        """Render AI match score visualization."""
        st.subheader("🎯 AI Match Score Breakdown")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                "Overall AI Score",
                f"{match_score.overall_score:.1f}%"
            )
        
        with col2:
            # Create radar chart for score breakdown
            categories = ['Technical Skills', 'Experience', 'Education', 'Soft Skills', 'ATS Score']
            scores = [
                match_score.technical_skills_score,
                match_score.experience_score,
                match_score.education_score,
                match_score.soft_skills_score,
                match_score.ats_score
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name='Your Score'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Score Breakdown",
                paper_bgcolor=current_theme['bg_color'],
                plot_bgcolor=current_theme['bg_color'],
                font=dict(color=current_theme['text_color'])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_skill_gap_analysis(self, skill_gap):
        """Render skill gap analysis."""
        st.subheader("🎯 Skill Gap Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ❌ Missing Skills")
            if skill_gap.missing_skills:
                for skill in skill_gap.missing_skills[:10]:
                    st.markdown(f"- {skill}")
            else:
                st.success("No major skills missing!")
        
        with col2:
            st.markdown("### 🔥 Priority Skills to Develop")
            if skill_gap.priority_skills:
                for i, skill in enumerate(skill_gap.priority_skills[:5], 1):
                    st.markdown(f"{i}. **{skill}**")
        
        # Recommendations
        if hasattr(skill_gap, 'recommended_learning_path') and skill_gap.recommended_learning_path:
            st.markdown("### 💡 Skill Development Recommendations")
            for rec in skill_gap.recommended_learning_path[:5]:
                st.markdown(f'<div class="recommendation">📚 {rec}</div>', unsafe_allow_html=True)
    
    def render_salary_estimation(self, salary_estimation):
        """Render salary estimation."""
        st.subheader("💰 Salary Estimation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Estimated Range (Min)",
                f"${salary_estimation.estimated_range_min:,.0f}"
            )
        
        with col2:
            st.metric(
                "Market Average",
                f"${salary_estimation.market_average:,.0f}"
            )
        
        with col3:
            st.metric(
                "Estimated Range (Max)",
                f"${salary_estimation.estimated_range_max:,.0f}"
            )
        
        # Salary breakdown chart
        if hasattr(salary_estimation, 'factors_affecting_salary') and salary_estimation.factors_affecting_salary:
            st.markdown("### 📈 Salary Factors")
            for i, factor in enumerate(salary_estimation.factors_affecting_salary[:5], 1):
                st.markdown(f"{i}. {factor}")
        
        # Market insights
        if hasattr(salary_estimation, 'improvement_potential') and salary_estimation.improvement_potential:
            st.markdown("### 📈 Improvement Potential")
            st.info(f"💡 {salary_estimation.improvement_potential}")
    
    def render_ats_optimization(self, ats_optimization):
        """Render ATS optimization suggestions."""
        st.subheader("🤖 ATS Optimization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # ATS Score gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = ats_optimization.ats_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "ATS Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor=current_theme['bg_color'],
                plot_bgcolor=current_theme['bg_color'],
                font=dict(color=current_theme['text_color']),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🔧 ATS Optimization Suggestions")
            if hasattr(ats_optimization, 'formatting_improvements') and ats_optimization.formatting_improvements:
                for suggestion in ats_optimization.formatting_improvements[:5]:
                    st.markdown(f'<div class="ai-insight">🔧 {suggestion}</div>', unsafe_allow_html=True)
        
        # Keyword optimization
        if hasattr(ats_optimization, 'missing_ats_keywords') and ats_optimization.missing_ats_keywords:
            st.markdown("### 🔑 Missing ATS Keywords")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Add these keywords:**")
                for keyword in ats_optimization.missing_ats_keywords[:10]:
                    st.markdown(f"- {keyword}")
            
            with col2:
                if hasattr(ats_optimization, 'keyword_suggestions') and ats_optimization.keyword_suggestions:
                    st.markdown("**Keyword suggestions:**")
                    for keyword in ats_optimization.keyword_suggestions[:10]:
                        st.markdown(f"- {keyword}")
    
    def render_ai_feedback(self, ai_analysis):
        """Render AI feedback and recommendations."""
        st.subheader("💬 AI Feedback & Recommendations")
        
        # Detailed feedback
        if 'ai_feedback' in ai_analysis and ai_analysis['ai_feedback']:
            feedback_data = ai_analysis['ai_feedback']
            
            if isinstance(feedback_data, dict) and 'detailed_feedback' in feedback_data:
                st.markdown("### 📝 Detailed Analysis")
                # Use st.markdown to properly render the markdown formatting
                st.markdown(feedback_data['detailed_feedback'])
            elif isinstance(feedback_data, str):
                st.markdown("### 📝 Detailed Analysis") 
                st.markdown(feedback_data)
        
        # Improvement plan
        if 'improvement_plan' in ai_analysis and ai_analysis['improvement_plan']:
            st.markdown("### 📋 Improvement Plan")
            for i, item in enumerate(ai_analysis['improvement_plan'][:8], 1):
                st.markdown(f"**{i}.** {item}")
        
        # Next steps
        if 'next_steps' in ai_analysis and ai_analysis['next_steps']:
            st.markdown("### 🚀 Immediate Next Steps")
            for i, step in enumerate(ai_analysis['next_steps'][:5], 1):
                st.markdown(f"**{i}.** {step}")
    
    def render_visualizations_tab(self):
        """Render visualizations tab."""
        st.header("📊 Analysis Visualizations")
        
        if not hasattr(st.session_state, 'analysis_results') or not st.session_state.analysis_results:
            st.info("🔍 Run an analysis first to see visualizations here!")
            return
        
        results = st.session_state.analysis_results
        
        # AI analysis charts only
        if 'ai_analysis' in results and results['ai_analysis']:
            self.render_ai_visualizations(results['ai_analysis'])
        else:
            st.warning("🤖 AI analysis results not available. Please ensure AI analysis completed successfully.")
    
    def render_ai_visualizations(self, ai_analysis):
        """Render AI analysis visualizations."""
        st.subheader("🤖 AI Analysis Visualizations")
        
        # AI Score Breakdown
        if 'ai_match_score' in ai_analysis:
            match_score = ai_analysis['ai_match_score']
            
            # Create breakdown chart
            breakdown_data = {
                'Category': ['Technical Skills', 'Experience', 'Education', 'Soft Skills', 'ATS Score'],
                'Score': [
                    match_score.technical_skills_score,
                    match_score.experience_score, 
                    match_score.education_score,
                    match_score.soft_skills_score,
                    match_score.ats_score
                ]
            }
            
            fig = px.bar(
                breakdown_data,
                x='Category',
                y='Score',
                title="AI Analysis Score Breakdown",
                color='Score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                showlegend=False,
                paper_bgcolor=current_theme['bg_color'],
                plot_bgcolor=current_theme['bg_color'],
                font=dict(color=current_theme['text_color']),
                xaxis=dict(
                    gridcolor=current_theme['border_color'],
                    color=current_theme['text_color']
                ),
                yaxis=dict(
                    gridcolor=current_theme['border_color'],
                    color=current_theme['text_color']
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_results_tab(self):
        """Render results tab."""
        st.header("💾 Analysis Results")
        
        if not hasattr(st.session_state, 'analysis_results') or not st.session_state.analysis_results:
            st.info("🔍 Run an analysis first to see results here!")
            return
        
        results = st.session_state.analysis_results
        
        # Summary
        summary = results.get('summary', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            ai_score = summary.get('ai_overall_score', 0)
            st.metric("AI Analysis Score", f"{ai_score:.1f}%")
        
        with col2:
            recommendations_count = len(summary.get('recommendations', []))
            st.metric("Recommendations", f"{recommendations_count} items")
        
        # Download results
        if st.button("📥 Download Results (JSON)"):
            results_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download Analysis Results",
                data=results_json,
                file_name="resume_analysis_results.json",
                mime="application/json"
            )
        
        # Raw results (collapsible)
        with st.expander("🔍 View Raw Results"):
            st.json(results)
    
    def extract_text_from_upload(self, uploaded_file) -> str:
        """Extract text from uploaded file."""
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text
            text = self.text_extractor.extract_text(temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            return text
        except Exception as e:
            st.error(f"Error extracting text: {str(e)}")
            return ""
    
    def perform_analysis(self):
        """Perform AI-powered analysis only."""
        if not hasattr(st.session_state, 'resume_text') or not hasattr(st.session_state, 'job_text'):
            st.error("❌ Please provide both resume and job description!")
            return
        
        if not st.session_state.resume_text.strip() or not st.session_state.job_text.strip():
            st.error("❌ Both resume and job description must contain text!")
            return

        # Check if AI is available
        if not self.ai_available:
            st.error("❌ AI analysis is required but unavailable. Please configure your OpenRouter API key.")
            return
        
        with st.spinner("🤖 Performing AI-powered analysis..."):
            try:
                # Prepare context
                context = {}
                if hasattr(st.session_state, 'location') and st.session_state.location:
                    context['location'] = st.session_state.location
                if hasattr(st.session_state, 'industry') and st.session_state.industry:
                    context['industry'] = st.session_state.industry
                
                # AI analysis only
                ai_results = self.perform_ai_analysis(
                    st.session_state.resume_text,
                    st.session_state.job_text,
                    context
                )
                
                # Store results
                st.session_state.analysis_results = {
                    'metadata': {
                        'ai_enhanced': True,
                        'context': context
                    },
                    'ai_analysis': ai_results,
                    'summary': self.create_analysis_summary(ai_results)
                }
                
                st.success("✅ AI analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.error(f"Debug info: {traceback.format_exc()}")
    
    def perform_ai_analysis(self, resume_text: str, job_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI-powered analysis."""
        try:
            ai_analysis = self.ai_analyzer.analyze_resume_comprehensive(
                resume_text, job_text, context
            )
            
            return {
                'ai_match_score': ai_analysis.match_score,
                'skill_gap_analysis': ai_analysis.skill_gap,
                'salary_estimation': ai_analysis.salary_estimation,
                'ats_optimization': ai_analysis.ats_optimization,
                'ai_feedback': ai_analysis.detailed_feedback,
                'improvement_plan': ai_analysis.improvement_plan,
                'next_steps': ai_analysis.next_steps
            }
        except Exception as e:
            st.warning(f"AI analysis failed: {str(e)}")
            return {'ai_error': str(e)}
    
    def create_analysis_summary(self, ai_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create analysis summary from AI results only."""
        summary = {
            'recommendations': []
        }
        
        if ai_results and 'ai_match_score' in ai_results:
            ai_score = ai_results['ai_match_score']
            summary['ai_overall_score'] = ai_score.overall_score
            summary['ai_breakdown'] = {
                'technical_skills': ai_score.technical_skills_score,
                'experience': ai_score.experience_score,
                'education': ai_score.education_score,
                'soft_skills': ai_score.soft_skills_score,
                'ats_compatibility': ai_score.ats_score
            }
            
            if 'improvement_plan' in ai_results:
                summary['recommendations'].extend(ai_results['improvement_plan'])
        
        return summary
    
    def clear_session_state(self):
        """Clear analysis session state."""
        keys_to_clear = [
            'resume_text', 'job_text', 'analysis_results'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]


def main():
    """Main application entry point."""
    app = EnhancedStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
