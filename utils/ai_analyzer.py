"""
Advanced AI-Powered Resume Analyzer
===================================

Uses DeepSeek API through OpenRouter for intelligent resume analysis.
Provides comprehensive evaluation, skill gap analysis, ATS optimization,
salary estimation, and personalized improvement recommendations.

Copyright (c) 2025 SyazWak
Licensed under the MIT License - see LICENSE file for details.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

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
        env_model = os.getenv("AI_MODEL", os.getenv("DEEPSEEK_MODEL", "")).strip()
        if env_model:
            self.config.model = env_model

        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip().strip("'\"")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Load fallback models
        self.fallback_models = []
        fallback_models_str = os.getenv("FALLBACK_MODELS", "").strip()
        if fallback_models_str:
            self.fallback_models = [model.strip() for model in fallback_models_str.split(",")]

        # Add individual fallback model for backward compatibility
        fallback_model = os.getenv("FALLBACK_MODEL", "").strip()
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
            "X-Title": "AI Resume Analyzer",
        }

        # Track which model was last used successfully
        self.last_successful_model = self.config.model

        # Feature flags
        self.enable_salary = os.getenv('ENABLE_SALARY_ESTIMATION', 'true').lower() == 'true'
        self.enable_skill_gap = os.getenv('ENABLE_SKILL_GAP_ANALYSIS', 'true').lower() == 'true'
        self.enable_ats = os.getenv('ENABLE_ATS_OPTIMIZATION', 'true').lower() == 'true'

    def analyze_resume_comprehensive(
        self,
        resume_text: str,
        job_description: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> ComprehensiveAnalysis:
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

        # Match score (critical — failure is total)
        try:
            match_score = self._analyze_match_score(resume_text, job_description)
        except Exception as e:
            logger.error(f"Match score analysis failed: {e}")
            match_score = MatchScore(
                overall_score=50.0, technical_skills_score=50.0, experience_score=50.0,
                education_score=50.0, soft_skills_score=50.0, ats_score=50.0,
                explanation=f"Analysis error: {str(e)}"
            )

        # Skill gap analysis (optional via feature flag)
        skill_gap = SkillGap(
            missing_skills=["Analysis disabled or failed"],
            skill_level_gaps={},
            recommended_learning_path=[],
            priority_skills=[],
        )
        if self.enable_skill_gap:
            try:
                skill_gap = self._analyze_skill_gap(resume_text, job_description)
            except Exception as e:
                logger.error(f"Skill gap analysis failed: {e}")

        # ATS optimization (optional via feature flag)
        ats_optimization = ATSOptimization(
            ats_score=70.0, keyword_density_score=70.0, format_score=80.0,
            missing_ats_keywords=[], formatting_improvements=["Analysis disabled or failed"],
            keyword_suggestions=[],
        )
        if self.enable_ats:
            try:
                ats_optimization = self._analyze_ats_optimization(resume_text, job_description)
            except Exception as e:
                logger.error(f"ATS optimization analysis failed: {e}")

        # Salary estimation (optional via feature flag)
        salary_estimation = None
        if self.enable_salary:
            try:
                salary_estimation = self._estimate_salary(
                    resume_text, job_description, additional_context
                )
            except Exception as e:
                logger.error(f"Salary estimation failed: {e}")

        # Detailed feedback (non-critical)
        try:
            detailed_feedback = self._generate_detailed_feedback(
                resume_text, job_description, match_score, skill_gap
            )
        except Exception as e:
            logger.error(f"Detailed feedback generation failed: {e}")
            detailed_feedback = {"message": f"Feedback generation failed: {str(e)}"}

        # Create improvement plan (uses already-computed data)
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
            next_steps=next_steps,
        )

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
                    "max_tokens": self.config.max_tokens,
                }

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30,
                )

                logger.info(f"API Response Status for {model}: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    self.last_successful_model = model  # Track successful model
                    if i > 0:
                        logger.info(f"✅ Successfully switched to fallback model: {model}")
                    return result["choices"][0]["message"]["content"]

                elif response.status_code == 401:
                    error_msg = "❌ API authentication failed. Please check your OpenRouter API key and account status."
                    logger.error(f"{error_msg} Response: {response.text}")
                    # Auth errors are not recoverable with fallback models
                    return f"AUTH_ERROR: {error_msg}"

                elif response.status_code == 402:
                    error_msg = (
                        "💳 Insufficient credits. Please add credits to your OpenRouter account."
                    )
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
                        return "RATE_LIMIT_ERROR: Rate limit exceeded on all available models. Please try again later."

                else:
                    logger.error(
                        f"API request failed for {model}: {response.status_code} - {response.text}"
                    )

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
                    return "TIMEOUT_ERROR: API request timed out on all available models. Please try again."

            except requests.exceptions.ConnectionError:
                error_msg = f"🌐 Unable to connect to OpenRouter API for {model}."
                logger.error(error_msg)

                # If this is not the last model, try the next one
                if i < len(models_to_try) - 1:
                    logger.info(f"🔄 Connection error with {model}, trying next fallback model...")
                    continue
                else:
                    return "CONNECTION_ERROR: Unable to connect to OpenRouter API. Please check your internet connection."

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

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON from AI response with multiple fallback strategies.

        Args:
            response: Raw AI response text

        Returns:
            Parsed JSON dict or fallback with raw text
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(response.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Extract from markdown fences
        if "```json" in response:
            try:
                json_text = response.split("```json")[1].split("```")[0]
                return json.loads(json_text.strip())
            except (json.JSONDecodeError, IndexError):
                pass

        # Strategy 3: Find first { and last }
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 4: Clean common issues (trailing commas)
        try:
            import re
            cleaned = re.sub(r',\s*([}\]])', r'\1', response)
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(cleaned[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 5: Fallback — return raw text in explanation
        logger.warning(f"Failed to parse JSON response: {response[:200]}...")
        return {"explanation": response, "_parse_failed": True}

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model configuration and status."""
        return {
            "primary_model": self.config.model,
            "fallback_models": self.fallback_models,
            "last_successful_model": getattr(self, "last_successful_model", self.config.model),
            "total_models_available": 1 + len(self.fallback_models),
            "api_configured": bool(self.api_key),
        }

    def _analyze_match_score(self, resume_text: str, job_description: str) -> MatchScore:
        """Analyze detailed match score using AI."""
        system_prompt = """You are an expert HR recruiter and resume analyst. You MUST respond with valid JSON only — no markdown, no explanation text, no code fences.

Return a JSON object with this exact structure:
{
  "overall_score": <number 0-100>,
  "technical_skills_score": <number 0-100>,
  "experience_score": <number 0-100>,
  "education_score": <number 0-100>,
  "soft_skills_score": <number 0-100>,
  "ats_score": <number 0-100>,
  "explanation": "<string: 2-3 sentence scoring rationale>"
}

Scoring rubric:
- 90-100: Expert/Exceptional match
- 70-89: Strong/Proficient match
- 50-69: Moderate/Adequate match
- 30-49: Below average/Gaps present
- 0-29: Poor/Significant gaps"""

        prompt = f"""Analyze this resume against the job description. Score each category 0-100.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

Score each category fairly. Consider transferable skills and potential.
Respond with JSON only."""

        response = self._make_api_request(prompt, system_prompt)

        score_data = self._parse_json_response(response)
        if score_data.get("_parse_failed"):
            return MatchScore(
                overall_score=50.0, technical_skills_score=50.0, experience_score=50.0,
                education_score=50.0, soft_skills_score=50.0, ats_score=50.0,
                explanation=score_data.get("explanation", "Analysis completed")
            )

        return MatchScore(
            overall_score=score_data.get("overall_score", 0),
            technical_skills_score=score_data.get("technical_skills_score", 0),
            experience_score=score_data.get("experience_score", 0),
            education_score=score_data.get("education_score", 0),
            soft_skills_score=score_data.get("soft_skills_score", 0),
            ats_score=score_data.get("ats_score", 0),
            explanation=score_data.get("explanation", "Analysis completed"),
        )

    def _analyze_skill_gap(self, resume_text: str, job_description: str) -> SkillGap:
        """Analyze skill gaps using AI."""
        system_prompt = """You are a career development expert. You MUST respond with valid JSON only — no markdown, no explanation text, no code fences.

Return a JSON object with this exact structure:
{
  "missing_skills": ["skill1", "skill2"],
  "skill_level_gaps": {"skill_name": "gap description"},
  "recommended_learning_path": ["step1", "step2"],
  "priority_skills": ["skill1", "skill2"],
  "estimated_development_time": {"skill_name": "timeframe"}
}

For estimated_development_time, use:
- "1-2 weeks" for basic proficiency
- "1-3 months" for working knowledge
- "3-6 months" for competence
- "6+ months" for expertise"""

        prompt = f"""Analyze skill gaps between this resume and job description.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

Identify missing skills, skill level gaps, learning path, and priorities.
Respond with JSON only."""

        response = self._make_api_request(prompt, system_prompt)

        gap_data = self._parse_json_response(response)
        if gap_data.get("_parse_failed"):
            return SkillGap(
                missing_skills=["Unable to analyze"],
                skill_level_gaps={},
                recommended_learning_path=["Review job requirements", "Update skills"],
                priority_skills=[],
            )

        return SkillGap(
            missing_skills=gap_data.get("missing_skills", []),
            skill_level_gaps=gap_data.get("skill_level_gaps", {}),
            recommended_learning_path=gap_data.get("recommended_learning_path", []),
            priority_skills=gap_data.get("priority_skills", []),
        )

    def _analyze_ats_optimization(self, resume_text: str, job_description: str) -> ATSOptimization:
        """Analyze ATS optimization opportunities."""
        system_prompt = """You are an ATS (Applicant Tracking System) optimization expert. You MUST respond with valid JSON only — no markdown, no explanation text, no code fences.

Return a JSON object with this exact structure:
{
  "ats_score": <number 0-100>,
  "keyword_density_score": <number 0-100>,
  "format_score": <number 0-100>,
  "missing_ats_keywords": ["keyword1", "keyword2"],
  "formatting_improvements": ["improvement1", "improvement2"],
  "keyword_suggestions": ["suggestion1", "suggestion2"],
  "keyword_placement": {"keyword": "where to add it in the resume"}
}"""

        prompt = f"""Analyze this resume for ATS optimization against the job description.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

Evaluate ATS compatibility, keyword density, format, and missing keywords.
For each keyword suggestion, specify exactly where in the resume it should be placed.
Respond with JSON only."""

        response = self._make_api_request(prompt, system_prompt)

        ats_data = self._parse_json_response(response)
        if ats_data.get("_parse_failed"):
            return ATSOptimization(
                ats_score=75.0, keyword_density_score=70.0, format_score=80.0,
                missing_ats_keywords=[], formatting_improvements=["Review ATS compatibility"],
                keyword_suggestions=["Add job-relevant keywords"],
            )

        return ATSOptimization(
            ats_score=ats_data.get("ats_score", 70),
            keyword_density_score=ats_data.get("keyword_density_score", 70),
            format_score=ats_data.get("format_score", 80),
            missing_ats_keywords=ats_data.get("missing_ats_keywords", []),
            formatting_improvements=ats_data.get("formatting_improvements", []),
            keyword_suggestions=ats_data.get("keyword_suggestions", []),
        )

    def _estimate_salary(
        self, resume_text: str, job_description: str, context: Optional[Dict[str, Any]] = None
    ) -> SalaryEstimation:
        """Estimate salary range based on resume and job requirements."""
        system_prompt = """You are a compensation analyst. You MUST respond with valid JSON only — no markdown, no explanation text, no code fences.

Return a JSON object with this exact structure:
{
  "estimated_range_min": <number in USD>,
  "estimated_range_max": <number in USD>,
  "market_average": <number in USD>,
  "factors_affecting_salary": ["factor1", "factor2"],
  "improvement_potential": "<string: how to increase earning potential>"
}"""

        location = context.get("location", "General") if context else "General"
        industry = context.get("industry", "Technology") if context else "Technology"

        prompt = f"""Estimate salary for this candidate profile.

RESUME:
{resume_text[:2000]}

JOB DESCRIPTION:
{job_description[:1500]}

CONTEXT: Location: {location}, Industry: {industry}

Provide realistic USD annual salary estimates.
Respond with JSON only."""

        response = self._make_api_request(prompt, system_prompt)

        salary_data = self._parse_json_response(response)
        if salary_data.get("_parse_failed"):
            return SalaryEstimation(
                estimated_range_min=50000, estimated_range_max=80000, market_average=65000,
                factors_affecting_salary=["Experience level", "Technical skills"],
                improvement_potential="Develop additional skills and gain more experience",
            )

        return SalaryEstimation(
            estimated_range_min=salary_data.get("estimated_range_min", 50000),
            estimated_range_max=salary_data.get("estimated_range_max", 80000),
            market_average=salary_data.get("market_average", 65000),
            factors_affecting_salary=salary_data.get("factors_affecting_salary", []),
            improvement_potential=salary_data.get(
                "improvement_potential", "Continue skill development"
            ),
        )

    def _generate_detailed_feedback(
        self, resume_text: str, job_description: str, match_score: MatchScore, skill_gap: SkillGap
    ) -> Dict[str, Any]:
        """Generate detailed feedback using AI."""
        system_prompt = """You are an expert career coach and resume analyst. Provide detailed, actionable feedback using markdown formatting with clear section headers (##), bullet points, and bold text for emphasis."""

        prompt = f"""Analyze this resume against the job description and provide comprehensive feedback.

RESUME:
{resume_text[:2500]}

JOB DESCRIPTION:
{job_description[:1500]}

CURRENT ANALYSIS:
- Overall Match Score: {match_score.overall_score}%
- Technical Skills Score: {match_score.technical_skills_score}%
- Missing Skills: {", ".join(skill_gap.missing_skills[:5]) if skill_gap.missing_skills else "None identified"}

Provide feedback in this structure:
## Resume Strengths
## Areas for Improvement
## Content Recommendations
## Formatting & Structure
## Next Steps

Be constructive, specific, and actionable."""

        response = self._make_api_request(prompt, system_prompt)

        return {
            "detailed_feedback": response,
            "timestamp": time.time(),
            "feedback_type": "ai_generated",
        }

    def _create_improvement_plan(
        self, match_score: MatchScore, skill_gap: SkillGap, ats_optimization: ATSOptimization
    ) -> List[str]:
        """Create prioritized improvement plan."""
        improvements = []

        # Priority based on scores
        if match_score.technical_skills_score < 70:
            improvements.extend(
                [
                    "Strengthen technical skills section with job-relevant technologies",
                    "Add specific project examples demonstrating required skills",
                ]
            )

        if match_score.experience_score < 60:
            improvements.extend(
                [
                    "Rewrite experience descriptions to highlight relevant achievements",
                    "Quantify accomplishments with metrics and results",
                ]
            )

        if ats_optimization.ats_score < 75:
            improvements.extend(
                [
                    "Optimize resume format for ATS compatibility",
                    "Incorporate more job-specific keywords naturally",
                ]
            )

        # Add skill gap improvements
        if skill_gap.priority_skills:
            improvements.append(f"Focus on developing: {', '.join(skill_gap.priority_skills[:3])}")

        return improvements[:8]  # Limit to top 8 improvements

    def _generate_next_steps(self, improvement_plan: List[str], skill_gap: SkillGap) -> List[str]:
        """Generate immediate next steps."""
        next_steps = [
            "Review and implement top 3 improvement suggestions",
            "Update resume with job-specific keywords",
            "Tailor cover letter to address skill gaps positively",
        ]

        if skill_gap.recommended_learning_path:
            next_steps.append(f"Start learning: {skill_gap.recommended_learning_path[0]}")

        next_steps.extend(
            [
                "Practice interview responses highlighting transferable skills",
                "Research company culture and values for better alignment",
                "Network with professionals in target industry",
            ]
        )

        return next_steps

    def _create_fallback_analysis(
        self, resume_text: str, job_description: str
    ) -> ComprehensiveAnalysis:
        """Create fallback analysis when AI is unavailable."""
        return ComprehensiveAnalysis(
            match_score=MatchScore(
                overall_score=60.0,
                technical_skills_score=60.0,
                experience_score=60.0,
                education_score=70.0,
                soft_skills_score=65.0,
                ats_score=70.0,
                explanation="AI analysis unavailable - basic assessment provided",
            ),
            skill_gap=SkillGap(
                missing_skills=["AI analysis required for detailed skill gap"],
                skill_level_gaps={},
                recommended_learning_path=["Set up AI API for detailed analysis"],
                priority_skills=[],
            ),
            salary_estimation=SalaryEstimation(
                estimated_range_min=50000,
                estimated_range_max=80000,
                market_average=65000,
                factors_affecting_salary=["Experience", "Skills", "Location"],
                improvement_potential="Enhance skills and experience",
            ),
            ats_optimization=ATSOptimization(
                ats_score=70.0,
                keyword_density_score=65.0,
                format_score=75.0,
                missing_ats_keywords=[],
                formatting_improvements=["Enable AI analysis for specific suggestions"],
                keyword_suggestions=[],
            ),
            detailed_feedback={"message": "Configure AI API for detailed feedback"},
            improvement_plan=["Set up DeepSeek API key", "Run comprehensive analysis"],
            next_steps=["Add API configuration", "Retry analysis"],
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
Missing Skills: {", ".join(analysis.skill_gap.missing_skills)}
Priority Skills: {", ".join(analysis.skill_gap.priority_skills)}

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
Missing Keywords: {", ".join(analysis.ats_optimization.missing_ats_keywords)}

🚀 IMPROVEMENT PLAN:
"""

        for i, improvement in enumerate(analysis.improvement_plan, 1):
            report += f"{i}. {improvement}\n"

        report += "\n📋 NEXT STEPS:\n"
        for i, step in enumerate(analysis.next_steps, 1):
            report += f"{i}. {step}\n"

        return report
