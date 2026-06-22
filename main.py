#!/usr/bin/env python3
"""
AI Resume Analyzer - Main CLI Application
==========================================

AI-powered command-line interface for resume analysis using DeepSeek.
Provides intelligent insights and recommendations.

Copyright (c) 2025 SyazWak
Licensed under the MIT License - see LICENSE file for details.

Usage:
    python main.py --resume path/to/resume.pdf --job_description path/to/job_desc.txt
    python main.py --resume resume.txt --job_description job.txt
    python main.py --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.ai_analyzer import AdvancedAIAnalyzer  # noqa: E402
from utils.text_extractor import TextExtractor  # noqa: E402
from utils.visualizer import ReportVisualizer  # noqa: E402


class AIResumeAnalyzer:
    """
    AI-powered Resume Analyzer using DeepSeek.
    """

    def __init__(self):
        """Initialize the AI-powered analyzer."""
        self.text_extractor = TextExtractor()
        self.visualizer = ReportVisualizer()

        # Initialize AI analyzer
        self.ai_analyzer = None
        self.ai_available = False
        try:
            self.ai_analyzer = AdvancedAIAnalyzer()
            print("🤖 AI-powered analysis enabled!")
            self.ai_available = True
        except Exception as e:
            print(f"⚠️ AI analysis unavailable: {e}")
            self.ai_available = False

    def analyze_resume(
        self,
        resume_path: str,
        job_description_path: str,
        output_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform AI-powered resume analysis using DeepSeek.

        Args:
            resume_path (str): Path to resume file
            job_description_path (str): Path to job description file
            output_path (Optional[str]): Path to save results
            context (Optional[Dict]): Additional context (location, industry, etc.)

        Returns:
            Dict[str, Any]: AI analysis results
        """
        print("🚀 Starting AI Resume Analysis...")

        if not self.ai_available:
            return self._create_error_result(
                "AI analysis is not available. Please configure your OpenRouter API key."
            )

        try:
            # Step 1: Extract text from files
            print("📄 Extracting text from files...")
            resume_text = self._extract_text_safely(resume_path, "resume")
            job_text = self._extract_text_safely(job_description_path, "job description")

            if not resume_text or not job_text:
                return self._create_error_result("Failed to extract text from one or both files")

            # Step 2: AI-powered analysis
            print("🤖 Performing AI-powered analysis...")
            ai_analysis = self.ai_analyzer.analyze_resume_comprehensive(
                resume_text, job_text, context
            )
            ai_results = {
                "ai_match_score": ai_analysis.match_score,
                "skill_gap_analysis": ai_analysis.skill_gap,
                "salary_estimation": ai_analysis.salary_estimation,
                "ats_optimization": ai_analysis.ats_optimization,
                "ai_feedback": ai_analysis.detailed_feedback,
                "improvement_plan": ai_analysis.improvement_plan,
                "next_steps": ai_analysis.next_steps,
            }
            print("✨ AI analysis completed!")

            # Step 3: Prepare results
            results = {
                "metadata": {
                    "resume_path": resume_path,
                    "job_description_path": job_description_path,
                    "analysis_timestamp": self._get_timestamp(),
                    "ai_enhanced": True,
                    "context": context or {},
                },
                "ai_analysis": ai_results,
                "summary": self._create_analysis_summary(ai_results),
            }

            # Step 4: Save results if output path provided
            if output_path:
                print(f"💾 Saving results to {output_path}...")
                success = self._save_results(results, output_path)
                if success:
                    print("✅ Results saved successfully!")
                else:
                    print("⚠️ Failed to save results file")

            print("🎉 Analysis completed successfully!")
            return results

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return self._create_error_result(error_msg)

    def _create_analysis_summary(self, ai_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary from AI analysis results."""
        summary = {"recommendations": []}

        if ai_results and "ai_match_score" in ai_results:
            ai_score = ai_results["ai_match_score"]
            summary["ai_overall_score"] = ai_score.overall_score
            summary["ai_breakdown"] = {
                "technical_skills": ai_score.technical_skills_score,
                "experience": ai_score.experience_score,
                "education": ai_score.education_score,
                "soft_skills": ai_score.soft_skills_score,
                "ats_compatibility": ai_score.ats_score,
            }

            # Add recommendations from AI analysis
            if "improvement_plan" in ai_results:
                summary["recommendations"].extend(ai_results["improvement_plan"])

        return summary

    def _extract_text_safely(self, file_path: str, file_type: str) -> str:
        """Safely extract text from a file with error handling."""
        try:
            # Validate file first
            is_valid, error_msg = self.text_extractor.validate_file(file_path)
            if not is_valid:
                print(f"❌ {file_type.title()} file validation failed: {error_msg}")
                return ""

            # Extract text
            text = self.text_extractor.extract_text(file_path)
            if text.strip():
                print(f"✅ Successfully extracted text from {file_type}")
                return text
            else:
                print(f"⚠️ {file_type.title()} file appears to be empty")
                return ""

        except Exception as e:
            print(f"❌ Failed to extract text from {file_type}: {str(e)}")
            return ""

    def _save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """Save analysis results to file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            print(f"Failed to save results: {e}")
            return False

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error result."""
        return {
            "error": True,
            "error_message": error_message,
            "summary": {
                "traditional_score": 0,
                "ai_overall_score": 0,
                "recommendations": [f"❌ {error_message}"],
            },
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print analysis results in a formatted way."""
        if results.get("error"):
            print(f"\n❌ Error: {results.get('error_message', 'Unknown error')}")
            return

        print("\n" + "=" * 60)
        print("🎯 AI-ENHANCED RESUME ANALYSIS REPORT")
        print("=" * 60)

        summary = results.get("summary", {})

        # Display scores
        traditional_score = summary.get("traditional_score", 0)
        ai_score = summary.get("ai_overall_score", 0)

        if ai_score > 0:
            print("\n📊 OVERALL SCORES:")
            print(f"   Traditional Analysis: {traditional_score:.1f}%")
            print(f"   AI-Enhanced Score:    {ai_score:.1f}%")

            # Show AI breakdown
            ai_breakdown = summary.get("ai_breakdown", {})
            if ai_breakdown:
                print("\n🔍 AI DETAILED BREAKDOWN:")
                for category, score in ai_breakdown.items():
                    print(f"   {category.replace('_', ' ').title()}: {score:.1f}%")
        else:
            print(f"\n📊 TRADITIONAL ANALYSIS SCORE: {traditional_score:.1f}%")

        # Show AI-specific insights
        ai_analysis = results.get("ai_analysis", {})
        if ai_analysis and "skill_gap_analysis" in ai_analysis:
            skill_gap = ai_analysis["skill_gap_analysis"]
            print("\n🎯 SKILL GAP ANALYSIS:")
            print(f"   Missing Skills: {', '.join(skill_gap.missing_skills[:5])}")
            print(f"   Priority Skills: {', '.join(skill_gap.priority_skills[:3])}")

        # Show salary estimation
        if ai_analysis and "salary_estimation" in ai_analysis and ai_analysis["salary_estimation"]:
            salary = ai_analysis["salary_estimation"]
            print("\n💰 SALARY ESTIMATION:")
            print(
                f"   Range: ${salary.estimated_range_min:,.0f} - ${salary.estimated_range_max:,.0f}"
            )
            print(f"   Market Average: ${salary.market_average:,.0f}")

        # Show ATS optimization
        if ai_analysis and "ats_optimization" in ai_analysis:
            ats = ai_analysis["ats_optimization"]
            print(f"\n🤖 ATS OPTIMIZATION SCORE: {ats.ats_score:.1f}%")

        # Show recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print("\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")

        print("=" * 60)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Resume Analyzer - Compare resumes with job descriptions using DeepSeek AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --resume resume.pdf --job_description job.txt
  python main.py -r resume.txt -j job_desc.txt --output results.json --location "San Francisco" --industry "Technology"
  python main.py --resume resume.pdf --job_description job.txt --traditional-only
        """,
    )

    parser.add_argument(
        "--resume", "-r", type=str, required=True, help="Path to the resume file (PDF or TXT)"
    )

    parser.add_argument(
        "--job_description",
        "-j",
        type=str,
        required=True,
        help="Path to the job description file (PDF or TXT)",
    )

    parser.add_argument(
        "--output", "-o", type=str, help="Path to save the analysis results (JSON format)"
    )

    parser.add_argument(
        "--location", type=str, help='Location for salary estimation (e.g., "San Francisco, CA")'
    )

    parser.add_argument(
        "--industry", type=str, help='Industry for context (e.g., "Technology", "Finance")'
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    return parser


def main():
    """Main entry point for the CLI application."""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Validate input files exist
    resume_path = Path(args.resume)
    job_path = Path(args.job_description)

    if not resume_path.exists():
        print(f"❌ Error: Resume file not found: {resume_path}")
        sys.exit(1)

    if not job_path.exists():
        print(f"❌ Error: Job description file not found: {job_path}")
        sys.exit(1)

    # Prepare context
    context = {}
    if args.location:
        context["location"] = args.location
    if args.industry:
        context["industry"] = args.industry

    # Initialize analyzer
    print("🤖 Initializing AI Resume Analyzer...")
    analyzer = AIResumeAnalyzer()

    # Perform analysis
    results = analyzer.analyze_resume(
        str(resume_path), str(job_path), args.output, context if context else None
    )

    # Print results
    analyzer.print_results(results)

    # Exit with appropriate code
    if results.get("error"):
        sys.exit(1)
    else:
        final_score = results.get("summary", {}).get("ai_overall_score", 0)
        print(f"\n🎯 Analysis complete! AI Score: {final_score:.1f}%")

        # Show next steps if available
        ai_analysis = results.get("ai_analysis", {})
        if ai_analysis and "next_steps" in ai_analysis:
            next_steps = ai_analysis["next_steps"]
            print("\n📋 IMMEDIATE NEXT STEPS:")
            for i, step in enumerate(next_steps[:3], 1):
                print(f"   {i}. {step}")

        sys.exit(0)


if __name__ == "__main__":
    main()
