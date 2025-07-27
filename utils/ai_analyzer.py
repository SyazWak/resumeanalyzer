"""
Advanced AI-Powered Resume Analyzer
===================================

Uses DeepSeek API through OpenRouter for intelligent resume analysis.
Provides comprehensive evaluation, skill gap analysis, ATS optimization,
salary estimation, and personalized improvement recommendations.

Copyright (c) 2025 SyazWak
Licensed under the MIT License - see LICENSE file for details.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for AI analysis."""
    model: str = "deepseek/deepseek-chat-v3-0324:free"
    temperature: float = 0.7
    max_tokens: int = 2000
    enable_detailed_analysis: bool = True
    enable_salary_estimation: bool = True
    enable_skill_gap_analysis: bool = True
    enable_ats_optimization: bool = True
    enable_improvement_suggestions: bool = True


@dataclass
class MatchScore:
    """Detailed match scoring."""
    overall_score: float
    technical_skills_score: float
    experience_score: float
    education_score: float
    soft_skills_score: float
    ats_score: float
    explanation: str


@dataclass
class SkillGap:
    """Skill gap analysis result."""
    missing_skills: List[str]
    skill_level_gaps: Dict[str, str]
    recommended_learning_path: List[str]
    priority_skills: List[str]


@dataclass
class SalaryEstimation:
    """Salary estimation result."""
    estimated_range_min: float
    estimated_range_max: float
    market_average: float
    factors_affecting_salary: List[str]
    improvement_potential: str


@dataclass
class ATSOptimization:
    """ATS optimization suggestions."""
    ats_score: float
    keyword_density_score: float
    format_score: float
    missing_ats_keywords: List[str]
    formatting_improvements: List[str]
    keyword_suggestions: List[str]


@dataclass
class ComprehensiveAnalysis:
    """Complete analysis result."""
    match_score: MatchScore
    skill_gap: SkillGap
    salary_estimation: Optional[SalaryEstimation]
    ats_optimization: ATSOptimization
    detailed_feedback: Dict[str, Any]
    improvement_plan: List[str]
    next_steps: List[str]


class AdvancedAIAnalyzer:
    """
    Advanced AI-powered resume analyzer using DeepSeek API.
    
    Features:
    - Intelligent match scoring with detailed breakdown
    - Skill gap analysis with learning recommendations
    - ATS optimization suggestions
    - Salary estimation based on market data
    - Personalized improvement plans
    - Industry-specific insights
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the AI analyzer."""
        self.config = config or AnalysisConfig()
        
        # Override model with environment variable if available
        env_model = os.getenv('DEEPSEEK_MODEL', '').strip()
        if env_model:
            self.config.model = env_model
        
        self.api_key = os.getenv('OPENROUTER_API_KEY', '').strip().strip("'\"")
        self.base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        
        # Load fallback models
        self.fallback_models = []
        fallback_models_str = os.getenv('FALLBACK_MODELS', '').strip()
        if fallback_models_str:
            self.fallback_models = [model.strip() for model in fallback_models_str.split(',')]
        
        # Add individual fallback model for backward compatibility
        fallback_model = os.getenv('FALLBACK_MODEL', '').strip()
        if fallback_model and fallback_model not in self.fallback_models:
            self.fallback_models.append(fallback_model)
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. AI analysis will be limited.")
        else:
            logger.info(f"API key loaded successfully (length: {len(self.api_key)})")
            logger.info(f"Primary model: {self.config.model}")
            if self.fallback_models:
                logger.info(f"Fallback models: {', '.join(self.fallback_models)}")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ai-resume-analyzer",
            "X-Title": "AI Resume Analyzer"
        }
        
        # Track which model was last used successfully
        self.last_successful_model = self.config.model
    
    def analyze_resume_comprehensive(self, 
                                   resume_text: str, 
                                   job_description: str,
                                   additional_context: Optional[Dict[str, Any]] = None) -> ComprehensiveAnalysis:
        """
        Perform comprehensive AI-powered resume analysis.
        
        Args:
            resume_text (str): The resume content
            job_description (str): The job description
            additional_context (Dict): Additional context like industry, location, etc.
            
        Returns:
            ComprehensiveAnalysis: Complete analysis results
        """
        logger.info("Starting comprehensive AI analysis...")
        
        try:
            # Get detailed match analysis
            match_score = self._analyze_match_score(resume_text, job_description)
            
            # Analyze skill gaps
            skill_gap = self._analyze_skill_gap(resume_text, job_description)
            
            # ATS optimization
            ats_optimization = self._analyze_ats_optimization(resume_text, job_description)
            
            # Salary estimation (if enabled)
            salary_estimation = None
            if self.config.enable_salary_estimation:
                salary_estimation = self._estimate_salary(resume_text, job_description, additional_context)
            
            # Get detailed feedback
            detailed_feedback = self._generate_detailed_feedback(
                resume_text, job_description, match_score, skill_gap
            )
            
            # Create improvement plan
            improvement_plan = self._create_improvement_plan(
                match_score, skill_gap, ats_optimization
            )
            
            # Generate next steps
            next_steps = self._generate_next_steps(improvement_plan, skill_gap)
            
            return ComprehensiveAnalysis(
                match_score=match_score,
                skill_gap=skill_gap,
                salary_estimation=salary_estimation,
                ats_optimization=ats_optimization,
                detailed_feedback=detailed_feedback,
                improvement_plan=improvement_plan,
                next_steps=next_steps
            )
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._create_fallback_analysis(resume_text, job_description)
    
    def _make_api_request(self, prompt: str, system_prompt: str = "") -> str:
        """Make a request to the AI API with fallback model support."""
        if not self.api_key:
            return "AI analysis unavailable - no API key configured"
        
        # List of models to try (primary + fallbacks)
        models_to_try = [self.config.model] + self.fallback_models
        
        for i, model in enumerate(models_to_try):
            if i > 0:
                logger.info(f"Trying fallback model {i}: {model}")
            
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30
                )
                
                logger.info(f"API Response Status for {model}: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    self.last_successful_model = model  # Track successful model
                    if i > 0:
                        logger.info(f"✅ Successfully switched to fallback model: {model}")
                    return result['choices'][0]['message']['content']
                
                elif response.status_code == 401:
                    error_msg = "❌ API authentication failed. Please check your OpenRouter API key and account status."
                    logger.error(f"{error_msg} Response: {response.text}")
                    # Auth errors are not recoverable with fallback models
                    return f"AUTH_ERROR: {error_msg}"
                
                elif response.status_code == 402:
                    error_msg = "💳 Insufficient credits. Please add credits to your OpenRouter account."
                    logger.error(f"{error_msg} Response: {response.text}")
                    # Credit errors are not recoverable with fallback models
                    return f"CREDIT_ERROR: {error_msg}"
                
                elif response.status_code == 429:
                    error_msg = f"⏰ Rate limit exceeded for {model}."
                    logger.warning(f"{error_msg} Response: {response.text}")
                    
                    # If this is not the last model, try the next one
                    if i < len(models_to_try) - 1:
                        logger.info(f"🔄 Rate limited on {model}, trying next fallback model...")
                        continue
                    else:
                        # All models exhausted
                        return f"RATE_LIMIT_ERROR: Rate limit exceeded on all available models. Please try again later."
                
                else:
                    logger.error(f"API request failed for {model}: {response.status_code} - {response.text}")
                    
                    # If this is not the last model, try the next one
                    if i < len(models_to_try) - 1:
                        logger.info(f"🔄 Error with {model}, trying next fallback model...")
                        continue
                    else:
                        return f"API Error {response.status_code}: Please check your OpenRouter account and API key."
                        
            except requests.exceptions.Timeout:
                error_msg = f"⏱️ API request timed out for {model}."
                logger.error(error_msg)
                
                # If this is not the last model, try the next one
                if i < len(models_to_try) - 1:
                    logger.info(f"🔄 Timeout with {model}, trying next fallback model...")
                    continue
                else:
                    return f"TIMEOUT_ERROR: API request timed out on all available models. Please try again."
            
            except requests.exceptions.ConnectionError:
                error_msg = f"🌐 Unable to connect to OpenRouter API for {model}."
                logger.error(error_msg)
                
                # If this is not the last model, try the next one
                if i < len(models_to_try) - 1:
                    logger.info(f"🔄 Connection error with {model}, trying next fallback model...")
                    continue
                else:
                    return f"CONNECTION_ERROR: Unable to connect to OpenRouter API. Please check your internet connection."
            
            except Exception as e:
                logger.error(f"API request error for {model}: {e}")
                
                # If this is not the last model, try the next one
                if i < len(models_to_try) - 1:
                    logger.info(f"🔄 Error with {model}, trying next fallback model...")
                    continue
                else:
                    return f"REQUEST_ERROR: {str(e)}"
        
        # This should not be reached, but just in case
        return "ERROR: All fallback models failed"
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model configuration and status."""
        return {
            "primary_model": self.config.model,
            "fallback_models": self.fallback_models,
            "last_successful_model": getattr(self, 'last_successful_model', self.config.model),
            "total_models_available": 1 + len(self.fallback_models),
            "api_configured": bool(self.api_key)
        }
    
    def _analyze_match_score(self, resume_text: str, job_description: str) -> MatchScore:
        """Analyze detailed match score using AI."""
        system_prompt = """You are an expert HR recruiter and resume analyst. Analyze the resume against the job description and provide detailed scoring.

Return your analysis as valid JSON with this exact structure:
{
  "overall_score": float (0-100),
  "technical_skills_score": float (0-100),
  "experience_score": float (0-100),
  "education_score": float (0-100),
  "soft_skills_score": float (0-100),
  "ats_score": float (0-100),
  "explanation": "detailed explanation of scoring rationale"
}"""
        
        prompt = f"""
Analyze this resume against the job description and provide detailed match scoring:

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

Provide detailed scoring breakdown considering:
1. Technical skills alignment
2. Experience relevance and level
3. Educational background match
4. Soft skills demonstration
5. ATS compatibility and keyword presence

Be thorough but fair in your assessment. Consider transferable skills and potential.
"""
        
        response = self._make_api_request(prompt, system_prompt)
        
        try:
            # Extract JSON from response
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0]
            else:
                json_text = response
                
            score_data = json.loads(json_text.strip())
            
            return MatchScore(
                overall_score=score_data.get('overall_score', 0),
                technical_skills_score=score_data.get('technical_skills_score', 0),
                experience_score=score_data.get('experience_score', 0),
                education_score=score_data.get('education_score', 0),
                soft_skills_score=score_data.get('soft_skills_score', 0),
                ats_score=score_data.get('ats_score', 0),
                explanation=score_data.get('explanation', 'Analysis completed')
            )
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response for match score")
            return MatchScore(
                overall_score=50.0,
                technical_skills_score=50.0,
                experience_score=50.0,
                education_score=50.0,
                soft_skills_score=50.0,
                ats_score=50.0,
                explanation=f"AI Analysis Result: {response[:500]}..."
            )
    
    def _analyze_skill_gap(self, resume_text: str, job_description: str) -> SkillGap:
        """Analyze skill gaps using AI."""
        system_prompt = """You are a career development expert. Analyze skill gaps between the resume and job requirements.

Return your analysis as valid JSON with this exact structure:
{
  "missing_skills": ["skill1", "skill2", ...],
  "skill_level_gaps": {"skill": "gap description", ...},
  "recommended_learning_path": ["step1", "step2", ...],
  "priority_skills": ["skill1", "skill2", ...]
}"""
        
        prompt = f"""
Analyze the skill gaps between this resume and job description:

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

Identify:
1. Missing technical and soft skills
2. Skills present but at insufficient level
3. Recommended learning path with specific resources
4. Priority skills to focus on first

Provide actionable, specific recommendations.
"""
        
        response = self._make_api_request(prompt, system_prompt)
        
        try:
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0]
            else:
                json_text = response
                
            gap_data = json.loads(json_text.strip())
            
            return SkillGap(
                missing_skills=gap_data.get('missing_skills', []),
                skill_level_gaps=gap_data.get('skill_level_gaps', {}),
                recommended_learning_path=gap_data.get('recommended_learning_path', []),
                priority_skills=gap_data.get('priority_skills', [])
            )
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response for skill gap")
            return SkillGap(
                missing_skills=["Unable to analyze"],
                skill_level_gaps={},
                recommended_learning_path=["Review job requirements", "Update skills"],
                priority_skills=[]
            )
    
    def _analyze_ats_optimization(self, resume_text: str, job_description: str) -> ATSOptimization:
        """Analyze ATS optimization opportunities."""
        system_prompt = """You are an ATS (Applicant Tracking System) optimization expert. Analyze how well the resume will perform in automated screening systems.

Return your analysis as valid JSON with this exact structure:
{
  "ats_score": float (0-100),
  "keyword_density_score": float (0-100),
  "format_score": float (0-100),
  "missing_ats_keywords": ["keyword1", "keyword2", ...],
  "formatting_improvements": ["improvement1", "improvement2", ...],
  "keyword_suggestions": ["suggestion1", "suggestion2", ...]
}"""
        
        prompt = f"""
Analyze this resume for ATS optimization against the job description:

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

Evaluate:
1. ATS compatibility score
2. Keyword density and relevance
3. Format optimization for parsing
4. Missing critical keywords from job description
5. Specific formatting improvements needed
6. Strategic keyword placement suggestions

Focus on making the resume ATS-friendly while maintaining readability.
"""
        
        response = self._make_api_request(prompt, system_prompt)
        
        try:
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0]
            else:
                json_text = response
                
            ats_data = json.loads(json_text.strip())
            
            return ATSOptimization(
                ats_score=ats_data.get('ats_score', 70),
                keyword_density_score=ats_data.get('keyword_density_score', 70),
                format_score=ats_data.get('format_score', 80),
                missing_ats_keywords=ats_data.get('missing_ats_keywords', []),
                formatting_improvements=ats_data.get('formatting_improvements', []),
                keyword_suggestions=ats_data.get('keyword_suggestions', [])
            )
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response for ATS optimization")
            return ATSOptimization(
                ats_score=75.0,
                keyword_density_score=70.0,
                format_score=80.0,
                missing_ats_keywords=[],
                formatting_improvements=["Review ATS compatibility"],
                keyword_suggestions=["Add job-relevant keywords"]
            )
    
    def _estimate_salary(self, 
                        resume_text: str, 
                        job_description: str, 
                        context: Optional[Dict[str, Any]] = None) -> SalaryEstimation:
        """Estimate salary range based on resume and job requirements."""
        system_prompt = """You are a compensation analyst. Estimate salary ranges based on the candidate's profile and job requirements.

Return your analysis as valid JSON with this exact structure:
{
  "estimated_range_min": float,
  "estimated_range_max": float,
  "market_average": float,
  "factors_affecting_salary": ["factor1", "factor2", ...],
  "improvement_potential": "description of how to increase earning potential"
}"""
        
        location = context.get('location', 'General') if context else 'General'
        industry = context.get('industry', 'Technology') if context else 'Technology'
        
        prompt = f"""
Estimate salary range for this candidate profile:

RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

CONTEXT:
Location: {location}
Industry: {industry}

Provide realistic salary estimates considering:
1. Candidate's experience level and skills
2. Job requirements and responsibilities
3. Industry standards and location
4. Market demand for these skills
5. Growth potential

Provide amounts in USD annually.
"""
        
        response = self._make_api_request(prompt, system_prompt)
        
        try:
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0]
            else:
                json_text = response
                
            salary_data = json.loads(json_text.strip())
            
            return SalaryEstimation(
                estimated_range_min=salary_data.get('estimated_range_min', 50000),
                estimated_range_max=salary_data.get('estimated_range_max', 80000),
                market_average=salary_data.get('market_average', 65000),
                factors_affecting_salary=salary_data.get('factors_affecting_salary', []),
                improvement_potential=salary_data.get('improvement_potential', 'Continue skill development')
            )
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response for salary estimation")
            return SalaryEstimation(
                estimated_range_min=50000,
                estimated_range_max=80000,
                market_average=65000,
                factors_affecting_salary=["Experience level", "Technical skills"],
                improvement_potential="Develop additional skills and gain more experience"
            )
    
    def _generate_detailed_feedback(self, 
                                  resume_text: str, 
                                  job_description: str,
                                  match_score: MatchScore,
                                  skill_gap: SkillGap) -> Dict[str, Any]:
        """Generate detailed feedback using AI."""
        system_prompt = """You are an expert career coach and resume analyst. Provide detailed, actionable feedback in a clear, well-structured format.

Use proper markdown formatting with:
- Clear section headers (##)
- Bullet points for lists
- **Bold** for emphasis
- Specific, actionable recommendations

Return structured feedback that is easy to read and implement."""
        
        prompt = f"""
Analyze this resume against the job description and provide comprehensive feedback:

RESUME:
{resume_text[:2500]}

JOB DESCRIPTION:
{job_description[:1500]}

CURRENT ANALYSIS:
- Overall Match Score: {match_score.overall_score}%
- Technical Skills Score: {match_score.technical_skills_score}%
- Experience Score: {match_score.experience_score}%
- Missing Skills: {', '.join(skill_gap.missing_skills[:5]) if skill_gap.missing_skills else 'None identified'}

Provide detailed feedback in the following structure:

## 📊 Resume Strengths
- List 3-5 key strengths and what makes them effective

## 🎯 Areas for Improvement
- Identify specific areas that need enhancement
- Explain why each area is important for this role

## 💡 Content Recommendations
- Specific suggestions for improving resume content
- Recommended keywords and phrases to include

## 🚀 Formatting & Structure
- Suggestions for better organization and presentation
- ATS optimization recommendations

## 📈 Next Steps
- Prioritized action items for immediate improvement
- Long-term development suggestions

Keep feedback constructive, specific, and actionable. Use professional language with clear examples.
"""
        
        response = self._make_api_request(prompt, system_prompt)
        
        return {
            "detailed_feedback": response,
            "timestamp": time.time(),
            "feedback_type": "ai_generated"
        }
    
    def _create_improvement_plan(self, 
                               match_score: MatchScore,
                               skill_gap: SkillGap,
                               ats_optimization: ATSOptimization) -> List[str]:
        """Create prioritized improvement plan."""
        improvements = []
        
        # Priority based on scores
        if match_score.technical_skills_score < 70:
            improvements.extend([
                "Strengthen technical skills section with job-relevant technologies",
                "Add specific project examples demonstrating required skills"
            ])
        
        if match_score.experience_score < 60:
            improvements.extend([
                "Rewrite experience descriptions to highlight relevant achievements",
                "Quantify accomplishments with metrics and results"
            ])
        
        if ats_optimization.ats_score < 75:
            improvements.extend([
                "Optimize resume format for ATS compatibility",
                "Incorporate more job-specific keywords naturally"
            ])
        
        # Add skill gap improvements
        if skill_gap.priority_skills:
            improvements.append(f"Focus on developing: {', '.join(skill_gap.priority_skills[:3])}")
        
        return improvements[:8]  # Limit to top 8 improvements
    
    def _generate_next_steps(self, improvement_plan: List[str], skill_gap: SkillGap) -> List[str]:
        """Generate immediate next steps."""
        next_steps = [
            "Review and implement top 3 improvement suggestions",
            "Update resume with job-specific keywords",
            "Tailor cover letter to address skill gaps positively"
        ]
        
        if skill_gap.recommended_learning_path:
            next_steps.append(f"Start learning: {skill_gap.recommended_learning_path[0]}")
        
        next_steps.extend([
            "Practice interview responses highlighting transferable skills",
            "Research company culture and values for better alignment",
            "Network with professionals in target industry"
        ])
        
        return next_steps
    
    def _create_fallback_analysis(self, resume_text: str, job_description: str) -> ComprehensiveAnalysis:
        """Create fallback analysis when AI is unavailable."""
        return ComprehensiveAnalysis(
            match_score=MatchScore(
                overall_score=60.0,
                technical_skills_score=60.0,
                experience_score=60.0,
                education_score=70.0,
                soft_skills_score=65.0,
                ats_score=70.0,
                explanation="AI analysis unavailable - basic assessment provided"
            ),
            skill_gap=SkillGap(
                missing_skills=["AI analysis required for detailed skill gap"],
                skill_level_gaps={},
                recommended_learning_path=["Set up AI API for detailed analysis"],
                priority_skills=[]
            ),
            salary_estimation=SalaryEstimation(
                estimated_range_min=50000,
                estimated_range_max=80000,
                market_average=65000,
                factors_affecting_salary=["Experience", "Skills", "Location"],
                improvement_potential="Enhance skills and experience"
            ),
            ats_optimization=ATSOptimization(
                ats_score=70.0,
                keyword_density_score=65.0,
                format_score=75.0,
                missing_ats_keywords=[],
                formatting_improvements=["Enable AI analysis for specific suggestions"],
                keyword_suggestions=[]
            ),
            detailed_feedback={"message": "Configure AI API for detailed feedback"},
            improvement_plan=["Set up DeepSeek API key", "Run comprehensive analysis"],
            next_steps=["Add API configuration", "Retry analysis"]
        )
    
    def export_analysis(self, analysis: ComprehensiveAnalysis, format: str = "json") -> str:
        """Export analysis results in specified format."""
        if format == "json":
            return json.dumps(asdict(analysis), indent=2, default=str)
        elif format == "text":
            return self._format_analysis_as_text(analysis)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _format_analysis_as_text(self, analysis: ComprehensiveAnalysis) -> str:
        """Format analysis as readable text report."""
        report = f"""
🎯 COMPREHENSIVE RESUME ANALYSIS REPORT
======================================

📊 MATCH SCORE BREAKDOWN:
Overall Match: {analysis.match_score.overall_score:.1f}%
- Technical Skills: {analysis.match_score.technical_skills_score:.1f}%
- Experience: {analysis.match_score.experience_score:.1f}%
- Education: {analysis.match_score.education_score:.1f}%
- Soft Skills: {analysis.match_score.soft_skills_score:.1f}%
- ATS Compatibility: {analysis.match_score.ats_score:.1f}%

{analysis.match_score.explanation}

🔍 SKILL GAP ANALYSIS:
Missing Skills: {', '.join(analysis.skill_gap.missing_skills)}
Priority Skills: {', '.join(analysis.skill_gap.priority_skills)}

📈 RECOMMENDED LEARNING PATH:
"""
        
        for i, step in enumerate(analysis.skill_gap.recommended_learning_path, 1):
            report += f"{i}. {step}\n"
        
        if analysis.salary_estimation:
            report += f"""
💰 SALARY ESTIMATION:
Estimated Range: ${analysis.salary_estimation.estimated_range_min:,.0f} - ${analysis.salary_estimation.estimated_range_max:,.0f}
Market Average: ${analysis.salary_estimation.market_average:,.0f}

"""
        
        report += f"""
🤖 ATS OPTIMIZATION (Score: {analysis.ats_optimization.ats_score:.1f}%):
Missing Keywords: {', '.join(analysis.ats_optimization.missing_ats_keywords)}

🚀 IMPROVEMENT PLAN:
"""
        
        for i, improvement in enumerate(analysis.improvement_plan, 1):
            report += f"{i}. {improvement}\n"
        
        report += "\n📋 NEXT STEPS:\n"
        for i, step in enumerate(analysis.next_steps, 1):
            report += f"{i}. {step}\n"
        
        return report
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Data class for AI analysis results."""
    overall_score: float
    match_percentage: float
    strengths: List[str]
    weaknesses: List[str]
    missing_skills: List[str]
    recommendations: List[str]
    keyword_analysis: Dict[str, Any]
    detailed_feedback: str
    improvement_priority: List[Dict[str, Any]]
    ats_score: float
    industry_fit: str


class AIResumeAnalyzer:
    """
    AI-powered resume analyzer using DeepSeek API.
    
    Provides intelligent analysis beyond simple keyword matching,
    including contextual understanding and personalized recommendations.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the AI Resume Analyzer.
        
        Args:
            api_key (Optional[str]): OpenRouter API key
            base_url (Optional[str]): OpenRouter API base URL
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = base_url or os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        self.model = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat-v3-0324:free')
        self.max_length = int(os.getenv('MAX_ANALYSIS_LENGTH', '4000'))
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.")
    
    def _make_api_call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Make a call to the DeepSeek API.
        
        Args:
            messages (List[Dict[str, str]]): Messages for the API
            
        Returns:
            Dict[str, Any]: API response
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': 0.7,
            'max_tokens': 2000,
            'top_p': 0.9
        }
        
        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            raise Exception(f"Failed to connect to OpenRouter API: {e}")
    
    def _truncate_text(self, text: str, max_length: int = None) -> str:
        """Truncate text to fit within API limits."""
        max_length = max_length or self.max_length
        if len(text) <= max_length:
            return text
        return text[:max_length] + "... [truncated]"
    
    def analyze_resume_comprehensive(self, resume_text: str, job_description: str) -> AnalysisResult:
        """
        Perform comprehensive AI-powered resume analysis.
        
        Args:
            resume_text (str): Resume content
            job_description (str): Job description content
            
        Returns:
            AnalysisResult: Comprehensive analysis results
        """
        try:
            # Step 1: Overall Analysis
            overall_analysis = self._analyze_overall_match(resume_text, job_description)
            
            # Step 2: Skills Gap Analysis
            skills_analysis = self._analyze_skills_gap(resume_text, job_description)
            
            # Step 3: ATS Optimization Analysis
            ats_analysis = self._analyze_ats_optimization(resume_text, job_description)
            
            # Step 4: Industry Fit Analysis
            industry_analysis = self._analyze_industry_fit(resume_text, job_description)
            
            # Step 5: Improvement Recommendations
            recommendations = self._generate_recommendations(
                resume_text, job_description, overall_analysis, skills_analysis
            )
            
            # Combine all analyses
            return self._compile_results(
                overall_analysis, skills_analysis, ats_analysis, 
                industry_analysis, recommendations
            )
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return self._create_fallback_result(str(e))
    
    def _analyze_overall_match(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze overall match between resume and job description."""
        
        prompt = f"""
        As an expert HR professional and resume analyst, analyze the match between this resume and job description.
        
        RESUME:
        {self._truncate_text(resume_text, 1500)}
        
        JOB DESCRIPTION:
        {self._truncate_text(job_description, 1000)}
        
        Provide analysis in this exact JSON format:
        {{
            "overall_score": 0.0-10.0,
            "match_percentage": 0-100,
            "strengths": ["strength1", "strength2", ...],
            "weaknesses": ["weakness1", "weakness2", ...],
            "detailed_feedback": "Comprehensive paragraph explaining the analysis",
            "key_alignments": ["alignment1", "alignment2", ...],
            "major_gaps": ["gap1", "gap2", ...]
        }}
        
        Focus on:
        - Relevant experience and skills
        - Educational background alignment
        - Project relevance
        - Technical skills match
        - Cultural fit indicators
        """
        
        messages = [
            {"role": "system", "content": "You are an expert resume analyst and HR professional."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_api_call(messages)
        return self._parse_json_response(response)
    
    def _analyze_skills_gap(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze skills gap and missing competencies."""
        
        prompt = f"""
        Analyze the skills gap between this resume and job requirements.
        
        RESUME:
        {self._truncate_text(resume_text, 1500)}
        
        JOB REQUIREMENTS:
        {self._truncate_text(job_description, 1000)}
        
        Provide analysis in this exact JSON format:
        {{
            "technical_skills": {{
                "present": ["skill1", "skill2", ...],
                "missing": ["skill1", "skill2", ...],
                "partially_present": ["skill1", "skill2", ...]
            }},
            "soft_skills": {{
                "present": ["skill1", "skill2", ...],
                "missing": ["skill1", "skill2", ...]
            }},
            "experience_level": {{
                "required": "junior/mid/senior",
                "candidate_level": "junior/mid/senior",
                "gap_analysis": "explanation"
            }},
            "priority_skills_to_develop": [
                {{"skill": "skill_name", "priority": "high/medium/low", "reason": "explanation"}},
                ...
            ]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a technical skills analyst specializing in career development."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_api_call(messages)
        return self._parse_json_response(response)
    
    def _analyze_ats_optimization(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze ATS (Applicant Tracking System) optimization."""
        
        prompt = f"""
        Analyze this resume for ATS (Applicant Tracking System) optimization against the job description.
        
        RESUME:
        {self._truncate_text(resume_text, 1500)}
        
        JOB DESCRIPTION:
        {self._truncate_text(job_description, 1000)}
        
        Provide analysis in this exact JSON format:
        {{
            "ats_score": 0-100,
            "keyword_optimization": {{
                "well_optimized_keywords": ["keyword1", "keyword2", ...],
                "missing_keywords": ["keyword1", "keyword2", ...],
                "keyword_density_issues": ["keyword1", "keyword2", ...]
            }},
            "formatting_issues": ["issue1", "issue2", ...],
            "ats_recommendations": ["recommendation1", "recommendation2", ...],
            "section_optimization": {{
                "skills_section": "good/needs_improvement/missing",
                "experience_section": "good/needs_improvement/missing",
                "education_section": "good/needs_improvement/missing"
            }}
        }}
        
        Focus on:
        - Keyword matching and density
        - Resume structure and sections
        - ATS-friendly formatting
        - Industry-specific terminology
        """
        
        messages = [
            {"role": "system", "content": "You are an ATS optimization expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_api_call(messages)
        return self._parse_json_response(response)
    
    def _analyze_industry_fit(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze industry fit and career trajectory."""
        
        prompt = f"""
        Analyze the candidate's industry fit and career trajectory for this position.
        
        RESUME:
        {self._truncate_text(resume_text, 1500)}
        
        JOB DESCRIPTION:
        {self._truncate_text(job_description, 1000)}
        
        Provide analysis in this exact JSON format:
        {{
            "industry_fit": "excellent/good/fair/poor",
            "career_progression": "on_track/needs_development/career_change",
            "industry_knowledge": {{
                "demonstrated_areas": ["area1", "area2", ...],
                "knowledge_gaps": ["gap1", "gap2", ...]
            }},
            "growth_potential": "high/medium/low",
            "cultural_fit_indicators": ["indicator1", "indicator2", ...],
            "red_flags": ["flag1", "flag2", ...]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a career development and industry analysis expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_api_call(messages)
        return self._parse_json_response(response)
    
    def _generate_recommendations(self, resume_text: str, job_description: str, 
                                overall_analysis: Dict, skills_analysis: Dict) -> Dict[str, Any]:
        """Generate personalized improvement recommendations."""
        
        prompt = f"""
        Based on the previous analyses, generate specific, actionable recommendations for improving this resume.
        
        CONTEXT:
        - Overall Score: {overall_analysis.get('overall_score', 'N/A')}
        - Main Gaps: {overall_analysis.get('major_gaps', [])}
        - Missing Skills: {skills_analysis.get('technical_skills', {}).get('missing', [])}
        
        RESUME:
        {self._truncate_text(resume_text, 1000)}
        
        JOB DESCRIPTION:
        {self._truncate_text(job_description, 800)}
        
        Provide recommendations in this exact JSON format:
        {{
            "immediate_actions": [
                {{"action": "specific action", "impact": "high/medium/low", "effort": "low/medium/high"}},
                ...
            ],
            "skill_development": [
                {{"skill": "skill_name", "learning_path": "how to learn it", "timeline": "timeframe"}},
                ...
            ],
            "resume_rewrites": [
                {{"section": "section_name", "current": "current text", "improved": "improved text"}},
                ...
            ],
            "interview_preparation": ["tip1", "tip2", ...],
            "portfolio_projects": [
                {{"project_idea": "project description", "skills_demonstrated": ["skill1", "skill2"]}},
                ...
            ]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a career coach and resume optimization expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_api_call(messages)
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON from API response."""
        try:
            content = response['choices'][0]['message']['content']
            # Extract JSON from the response (in case there's extra text)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            return json.loads(json_str)
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {response}")
            return {"error": "Failed to parse AI response", "raw_response": response}
    
    def _compile_results(self, overall_analysis: Dict, skills_analysis: Dict, 
                        ats_analysis: Dict, industry_analysis: Dict, 
                        recommendations: Dict) -> AnalysisResult:
        """Compile all analyses into a single result object."""
        
        return AnalysisResult(
            overall_score=overall_analysis.get('overall_score', 0),
            match_percentage=overall_analysis.get('match_percentage', 0),
            strengths=overall_analysis.get('strengths', []),
            weaknesses=overall_analysis.get('weaknesses', []),
            missing_skills=skills_analysis.get('technical_skills', {}).get('missing', []),
            recommendations=recommendations.get('immediate_actions', []),
            keyword_analysis=ats_analysis.get('keyword_optimization', {}),
            detailed_feedback=overall_analysis.get('detailed_feedback', ''),
            improvement_priority=recommendations.get('skill_development', []),
            ats_score=ats_analysis.get('ats_score', 0),
            industry_fit=industry_analysis.get('industry_fit', 'unknown')
        )
    
    def _create_fallback_result(self, error_message: str) -> AnalysisResult:
        """Create a fallback result when AI analysis fails."""
        return AnalysisResult(
            overall_score=0,
            match_percentage=0,
            strengths=["Unable to analyze due to technical error"],
            weaknesses=[f"Analysis failed: {error_message}"],
            missing_skills=[],
            recommendations=["Please try again or contact support"],
            keyword_analysis={},
            detailed_feedback="AI analysis temporarily unavailable",
            improvement_priority=[],
            ats_score=0,
            industry_fit="unknown"
        )
    
    def generate_cover_letter(self, resume_text: str, job_description: str, 
                            company_name: str = "", position: str = "") -> str:
        """Generate a personalized cover letter using AI."""
        
        prompt = f"""
        Write a compelling, personalized cover letter based on this resume and job description.
        
        RESUME:
        {self._truncate_text(resume_text, 1500)}
        
        JOB DESCRIPTION:
        {self._truncate_text(job_description, 1000)}
        
        COMPANY: {company_name or "the company"}
        POSITION: {position or "this position"}
        
        Write a professional cover letter that:
        - Is 3-4 paragraphs long
        - Highlights relevant experience from the resume
        - Addresses specific job requirements
        - Shows enthusiasm and cultural fit
        - Includes specific examples and achievements
        - Has a compelling opening and strong closing
        
        Format as a complete, ready-to-send cover letter.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert cover letter writer and career coach."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._make_api_call(messages)
            return response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Cover letter generation failed: {e}")
            return f"Failed to generate cover letter: {e}"
    
    def suggest_interview_questions(self, resume_text: str, job_description: str) -> List[str]:
        """Suggest potential interview questions based on the role and resume."""
        
        prompt = f"""
        Based on this resume and job description, suggest potential interview questions the candidate should prepare for.
        
        RESUME:
        {self._truncate_text(resume_text, 1500)}
        
        JOB DESCRIPTION:
        {self._truncate_text(job_description, 1000)}
        
        Provide 10-15 specific interview questions that:
        - Test relevant technical skills
        - Explore experience gaps
        - Assess cultural fit
        - Include behavioral questions
        - Cover both common and role-specific questions
        
        Format as a simple list, one question per line.
        """
        
        messages = [
            {"role": "system", "content": "You are an experienced technical interviewer and hiring manager."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._make_api_call(messages)
            content = response['choices'][0]['message']['content']
            # Split into lines and clean up
            questions = [q.strip() for q in content.split('\n') if q.strip() and '?' in q]
            return questions
        except Exception as e:
            logger.error(f"Interview questions generation failed: {e}")
            return [f"Failed to generate questions: {e}"]
    
    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            messages = [
                {"role": "user", "content": "Hello, please respond with 'Connection successful'"}
            ]
            response = self._make_api_call(messages)
            return "successful" in response['choices'][0]['message']['content'].lower()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
