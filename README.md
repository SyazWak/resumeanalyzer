# AI Resume Analyzer

An intelligent resume analysis system powered by AI that provides comprehensive insights for job applications.

## Features

- **Smart Match Scoring**: AI-driven compatibility assessment with detailed breakdown
- **Skill Gap Analysis**: Identifies missing skills and provides learning recommendations
- **ATS Optimization**: Suggests improvements for Applicant Tracking Systems
- **Salary Estimation**: AI-powered salary range prediction based on location and industry
- **Personalized Feedback**: Detailed improvement suggestions and next steps
- **Multi-Format Support**: Upload PDFs, TXT, or DOCX files
- **Dual Interface**: Command line and web interface options
- **Resume Rewrite**: AI-powered section-by-section improvement with diff view
- **PDF & DOCX Export**: Download improved resume in professional formats
- **Streamlit-native dark/light theme**: Automatic theme support via Streamlit's built-in settings
- **Resilient AI analysis with fallback model support**: Automatic failover to alternative AI models
- **Progress indicators during analysis**: Visual feedback with spinners and status updates

## Quick Start

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

## Project Structure

```
resumeanalyzer/
├── main.py                    # CLI application
├── app.py                     # Web interface (Streamlit)
├── utils/
│   ├── ai_analyzer.py         # AI analysis engine (OpenRouter)
│   ├── text_extractor.py      # File text extraction
│   ├── visualizer.py          # Charts and reports
│   ├── resume_writer.py       # Resume rewrite and export
│   └── results.py             # Shared result/summary assembly
├── .streamlit/
│   └── config.toml            # Native Streamlit theme defaults
├── static/
│   └── styles.css             # Minimal brand-polish CSS
├── tests/                     # Unit tests
├── pyproject.toml             # Project config
├── requirements.txt           # Dependencies
└── README.md
```

## Configuration

### Environment Variables (.env)
Copy `.env.example` to `.env` and adjust as needed. All of these are read by the analyzer:

| Variable | Purpose | Value in `.env.example` |
|---|---|---|
| `OPENROUTER_API_KEY` | Required. API key from OpenRouter | — |
| `AI_MODEL` | Primary chat model (`DEEPSEEK_MODEL` still works as a fallback name) | `deepseek/deepseek-chat-v3-0324:free` |
| `OPENROUTER_BASE_URL` | API endpoint | `https://openrouter.ai/api/v1` |
| `FALLBACK_MODELS` | Comma-separated models tried on rate limits/errors | — |
| `TEMPERATURE` / `MAX_TOKENS` / `API_TIMEOUT` | Generation and request settings | `0.1` / `4000` / `30` |
| `DEFAULT_LOCATION` / `DEFAULT_INDUSTRY` | Fallback context for salary estimation | `General` / `Technology` |
| `ENABLE_*` | Toggle detailed analysis, improvement suggestions, salary, skill gap, ATS | `true` |

### Theme
The app uses Streamlit's native theming. Defaults live in `.streamlit/config.toml` (dark by default); switch light/dark any time via the hamburger menu → **Settings → Theme**. `static/styles.css` only adds accent colors and spacing.

### Command Line Options
```bash
python main.py --resume resume.pdf --job_description job.txt [options]

Options:
  --location      Location for salary estimation
  --industry      Industry context
  --output, -o    Output JSON file path
  --verbose, -v   Enable detailed output
```

## Sample Output

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

## Troubleshooting

A built-in troubleshooting panel is available in the sidebar (click "Troubleshooting" expander) with common solutions.

**AI Analysis Not Working**
- Check your `.env` file has a valid, non-placeholder `OPENROUTER_API_KEY` (the app will warn in the sidebar/CLI if it is missing)
- Verify OpenRouter account has credits
- Check internet connectivity
- If an individual analysis step fails, the rest of the report still renders and the specific error is shown above the results and in the downloaded JSON (`summary.analysis_errors`)

**PDF Extraction Fails**
- Ensure PDF is not password protected
- Try using text input instead

**Installation Issues**
- Run `pip install -r requirements.txt`
- Ensure virtual environment is activated
- Check Python version (3.8+ required)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 SyazWak

## Acknowledgments

- **OpenRouter** for API infrastructure
- **Streamlit** for web interface

## Support

Email: muhdwaiz200@gmail.com