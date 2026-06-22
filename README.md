# 🤖 AI Resume Analyzer with DeepSeek

An intelligent resume analysis system powered by DeepSeek AI that provides comprehensive insights for job applications.

## ✨ Features

- **Smart Match Scoring**: AI-driven compatibility assessment with detailed breakdown
- **Skill Gap Analysis**: Identifies missing skills and provides learning recommendations
- **ATS Optimization**: Suggests improvements for Applicant Tracking Systems
- **Salary Estimation**: AI-powered salary range prediction based on location and industry
- **Personalized Feedback**: Detailed improvement suggestions and next steps
- **Multi-Format Support**: Upload PDFs, TXT, or DOCX files
- **Dual Interface**: Command line and web interface options
- **Resume Rewrite**: AI-powered section-by-section improvement with diff view
- **PDF & DOCX Export**: Download improved resume in professional formats

## 🚀 Quick Start

### 1. Setup
```bash
git clone <your-repo-url>
cd resumeanalyzer
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure API
```bash
cp .env.example .env
# Add your OpenRouter API key to .env file
OPENROUTER_API_KEY=your_api_key_here
```
Get your API key from: https://openrouter.ai/keys

### 3. Run
```bash
# Web Interface
streamlit run app.py

# Command Line
python main.py --resume resume.pdf --job_description job.txt
```

### 4. Resume Rewrite (after analysis)
- Click "Rewrite Resume" tab in the web interface
- Or use the CLI with `--resume` flag

## 📁 Project Structure

```
resumeanalyzer/
├── main.py                    # CLI application
├── app.py                     # Web interface (Streamlit)
├── utils/
│   ├── ai_analyzer.py         # AI analysis engine
│   ├── text_extractor.py      # File text extraction
│   ├── visualizer.py          # Charts and reports
│   └── resume_writer.py       # Resume rewrite and export
├── static/
│   └── styles.css             # Theme CSS
├── tests/                     # Unit tests
├── pyproject.toml             # Project config
├── requirements.txt           # Dependencies
└── README.md
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
OPENROUTER_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek/deepseek-chat-v3-0324:free
```

### Command Line Options
```bash
python main.py --resume resume.pdf --job_description job.txt [options]

Options:
  --location      Location for salary estimation
  --industry      Industry context
  --output, -o    Output JSON file path
  --verbose, -v   Enable detailed output
```

## 📊 Sample Output

```json
{
  "ai_analysis": {
    "ai_match_score": {
      "overall_score": 78.5,
      "technical_skills_score": 85.0,
      "experience_score": 75.0,
      "ats_score": 82.0
    },
    "skill_gap_analysis": {
      "missing_skills": ["React", "AWS", "Docker"],
      "priority_skills": ["React", "Cloud Computing"]
    },
    "salary_estimation": {
      "estimated_range_min": 90000,
      "estimated_range_max": 120000
    }
  }
}
```

## 🐛 Troubleshooting

**AI Analysis Not Working**
- Check your `.env` file has valid `OPENROUTER_API_KEY`
- Verify OpenRouter account has credits
- Check internet connectivity

**PDF Extraction Fails**
- Ensure PDF is not password protected
- Try using text input instead

**Installation Issues**
- Run `pip install -r requirements.txt`
- Ensure virtual environment is activated
- Check Python version (3.8+ required)

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 SyazWak

## �🙏 Acknowledgments

- **DeepSeek AI** for language model capabilities
- **OpenRouter** for API infrastructure
- **Streamlit** for web interface

## 📞 Support

📧 Email: muhdwaiz200@gmail.com