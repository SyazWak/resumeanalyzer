# 🤖 AI Resume Analyzer with DeepSeek

An intelligent resume analysis system powered by DeepSeek AI that provides comprehensive insights for job applications.

### ✨ Features

### 🎯 AI-Powered Analysis with DeepSeek
- **Smart Match Scoring**: AI-driven compatibility assessment with detailed breakdown
- **Skill Gap Analysis**: Identifies missing skills and provides learning recommendations
- **ATS Optimization**: Suggests improvements for Applicant Tracking Systems
- **Salary Estimation**: AI-powered salary range prediction based on location and industry
- **Personalized Feedback**: Detailed improvement suggestions and next steps
- **Market Insights**: Industry trends and competitive analysis

### 🌐 Interface Options
- **Command Line Interface**: Batch processing with AI analysis
- **Web Interface**: Interactive Streamlit dashboard with AI-powered insights
- **Flexible Input**: Upload PDFs/TXT/DOCX files or paste text directly

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd resumeanalyzer

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure DeepSeek API
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenRouter API key for DeepSeek access
OPENROUTER_API_KEY=your_api_key_here
```

Get your OpenRouter API key from: https://openrouter.ai/keys

### 3. Run Analysis

#### 🖥️ Command Line (AI Analysis)
```bash
# Basic AI analysis
python main.py --resume examples/waiz_resume.txt --job_description examples/junior_dev_job.txt

# With location and industry context
python main.py --resume resume.pdf --job_description job.txt --location "San Francisco, CA" --industry "Technology"
```

#### 🌐 Web Interface (AI-Powered)
```bash
# Start AI-powered web app
streamlit run app.py

# Or use VS Code task: "Start AI Resume Analyzer Web App"
```

#### 🔧 VS Code Integration
Use the built-in tasks:
- `Run AI Resume Analyzer CLI` - CLI analysis with AI
- `Start AI Resume Analyzer Web App` - Web interface with AI

## 📁 Project Structure

```
resumeanalyzer/
├── main.py                    # AI-powered CLI application
├── app.py                     # AI-powered web interface  
├── utils/
│   ├── ai_analyzer.py         # 🤖 Advanced AI analysis engine
│   ├── text_extractor.py      # PDF/TXT/DOCX text extraction
│   └── visualizer.py          # Chart and report generation
├── examples/
│   ├── waiz_resume.txt        # Sample resume
│   └── junior_dev_job.txt     # Sample job description
├── .env.example               # Environment configuration template
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required for AI features
OPENROUTER_API_KEY=your_openrouter_api_key_here
DEEPSEEK_MODEL=deepseek/deepseek-chat

# Optional settings
DEFAULT_LOCATION=San Francisco, CA
DEFAULT_INDUSTRY=Technology
MAX_TOKENS=4000
TEMPERATURE=0.1
```

### Command Line Options
```bash
Options:
  --resume, -r              Path to resume file (PDF/TXT)
  --job_description, -j     Path to job description file (PDF/TXT)
  --output, -o             Output path for results (JSON)
  --ai-analysis            Enable AI analysis (default: true)
  --traditional-only       Use only traditional NLP analysis
  --location               Location for salary estimation
  --industry               Industry context
  --verbose, -v            Enable verbose output
```

## 📊 Analysis Output

### AI Analysis Results
```json
{
  "ai_analysis": {
    "ai_match_score": {
      "overall_score": 78.5,
      "technical_skills_score": 85.0,
      "experience_score": 75.0,
      "education_score": 80.0,
      "soft_skills_score": 70.0,
      "ats_score": 82.0
    },
    "skill_gap_analysis": {
      "missing_skills": ["React", "AWS", "Docker"],
      "priority_skills": ["React", "Cloud Computing"],
      "recommendations": ["Take React course", "Get AWS certification"]
    },
    "salary_estimation": {
      "estimated_range_min": 90000,
      "estimated_range_max": 120000,
      "market_average": 105000
    },
    "ats_optimization": {
      "ats_score": 82.0,
      "suggestions": ["Add more keywords", "Use standard headings"]
    }
  }
}
```

## 🎯 Use Cases

### For Job Seekers
- **Resume Optimization**: Get AI-powered suggestions to improve your resume
- **Skill Gap Analysis**: Identify skills to develop for target roles
- **ATS Compatibility**: Ensure your resume passes automated screening
- **Salary Negotiation**: Get data-driven salary estimates

### For Recruiters
- **Candidate Screening**: Quickly assess resume-job fit with AI scoring
- **Bulk Analysis**: Process multiple resumes via CLI
- **Skill Assessment**: Identify candidate strengths and gaps

### For Career Coaches
- **Client Guidance**: Provide data-driven career advice
- **Market Analysis**: Understand skill demand and salary trends
- **Progress Tracking**: Monitor resume improvement over time

## 🛠️ Technical Features

### AI Capabilities
- **DeepSeek Integration**: Advanced language model for intelligent analysis
- **Context-Aware**: Considers location, industry, and role specifics
- **Multi-Dimensional Scoring**: Technical skills, experience, education, soft skills, ATS
- **Actionable Insights**: Specific recommendations with priority ranking

### File Processing
- **Multi-Format Support**: PDF, TXT, and DOCX file processing
- **Robust Extraction**: Multiple extraction methods with fallbacks
- **Text Cleaning**: Advanced preprocessing for better analysis
- **Error Handling**: Graceful handling of corrupted or unreadable files

## 🔍 Example Analysis

### Traditional vs AI Comparison
- **Traditional Score**: Not applicable (removed traditional analysis)
- **AI Score**: 78.5% (Comprehensive AI analysis)
- **Improvement**: Pure AI-driven assessment with detailed insights

### Sample AI Insights
- **Missing Skills**: React, AWS, Docker, Kubernetes
- **Priority Development**: Frontend frameworks, Cloud platforms
- **ATS Score**: 82% (Good, but can improve keyword density)
- **Salary Range**: $90K - $120K (San Francisco, Technology)

## 🚀 Advanced Usage

### Batch Processing
```bash
# Process multiple resumes
for resume in resumes/*.pdf; do
    python main_ai.py --resume "$resume" --job_description job.txt --output "results/$(basename "$resume" .pdf).json"
done
```

### API Integration
```python
from utils.ai_analyzer import AdvancedAIAnalyzer

analyzer = AdvancedAIAnalyzer()
results = analyzer.analyze_resume_comprehensive(
    resume_text, job_text, {"location": "SF", "industry": "Tech"}
)
```

### Custom Analysis
```python
# Configure analysis depth
config = AnalysisConfig(
    include_salary_estimation=True,
    include_skill_gap_analysis=True,
    analysis_depth="comprehensive"
)
results = analyzer.analyze_with_config(resume_text, job_text, config)
```

## 🐛 Troubleshooting

### Common Issues

1. **AI Analysis Unavailable**
   - Check your `.env` file has valid `OPENROUTER_API_KEY`
   - Verify your OpenRouter account has sufficient credits
   - Check internet connectivity

2. **PDF Extraction Fails**
   - Ensure PDF is not password protected
   - Try converting PDF to text manually
   - Use text input method instead

3. **Import Errors**
   - Run `pip install -r requirements.txt`
   - Ensure virtual environment is activated
   - Check Python version compatibility (3.8+)

### Debug Mode
```bash
# Enable verbose output
python main.py --resume resume.pdf --job_description job.txt --verbose

# Check analysis logs
streamlit run app.py --logger.level=debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **DeepSeek AI** for advanced language model capabilities
- **OpenRouter** for API infrastructure
- **spaCy** for minimal NLP processing
- **Streamlit** for web interface framework

## 📞 Support

- 📧 Email: [your-email@example.com]
- 💬 Issues: [GitHub Issues](https://github.com/your-username/resumeanalyzer/issues)
- 📖 Documentation: [Wiki](https://github.com/your-username/resumeanalyzer/wiki)
