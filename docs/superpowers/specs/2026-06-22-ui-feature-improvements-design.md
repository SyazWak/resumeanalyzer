# UI & Feature Improvements — Design Spec

**Date:** 2026-06-22
**Author:** SyazWak
**Status:** Draft

## Overview

This spec covers a comprehensive improvement to the AI Resume Analyzer: theme system redesign, model configuration generalization, full visual overhaul, error handling improvements, and analysis quality enhancements.

## Goals

1. Fix the broken dark/light theme system by switching to Streamlit-native theming
2. Generalize model configuration from DeepSeek-specific to provider-agnostic
3. Overhaul the UI for a professional, modern look
4. Improve error handling with user-friendly messages and retry logic
5. Improve analysis quality with better prompts and resilient JSON parsing

## Non-Goals

- Adding new features (batch analysis, user accounts, REST API)
- Migrating away from Streamlit
- Changing the core analysis architecture

---

## Phase 1: Theme System — Streamlit-Native

### 1.1 Create .streamlit/config.toml

Create `.streamlit/config.toml` with a dark mode default:

```toml
[theme]
base = "dark"
primaryColor = "#4dabf7"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#ffffff"
font = "sans serif"
```

### 1.2 Remove custom theme system from app.py

Delete the following from `app.py`:
- `get_system_theme()` function (lines 38-58)
- `sync_with_streamlit_theme()` function (lines 61-84)
- Theme initialization block (lines 87-92)
- `themes` dictionary (lines 100-131)
- `current_theme` assignment (line 133)
- CSS variable injection block (lines 136-157)
- Theme settings sidebar section (lines 211-261)

### 1.3 Replace static/styles.css

Replace the full CSS file with a minimal "brand polish" stylesheet that only adds:
- Accent colors for buttons and highlights
- Card spacing and border-radius
- Tab styling improvements
- No background or text color overrides (Streamlit handles that)

### 1.4 User theme switching

Users switch themes via Streamlit's built-in hamburger menu (top-right) → "Settings" → "Dark/Light mode". No custom toggle needed.

---

## Phase 2: Model Configuration — Generic Naming

### 2.1 Rename env var

**File:** `.env.example`

Rename `DEEPSEEK_MODEL` to `AI_MODEL`. Add backward compatibility in code.

### 2.2 Update AdvancedAIAnalyzer

**File:** `utils/ai_analyzer.py`

In `__init__()`, change:
```python
env_model = os.getenv('DEEPSEEK_MODEL', '').strip()
```
to:
```python
env_model = os.getenv('AI_MODEL', os.getenv('DEEPSEEK_MODEL', '')).strip()
```

This reads `AI_MODEL` first, falls back to `DEEPSEEK_MODEL` for existing users.

### 2.3 Update sidebar display

**File:** `app.py`

Change the sidebar model display from:
```python
primary_model = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat-v3-0324:free")
```
to:
```python
primary_model = os.getenv("AI_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat-v3-0324:free"))
```

### 2.4 Update .env.example

```bash
# AI Model Configuration
# Supported: Any OpenRouter model (deepseek/*, openai/*, anthropic/*, etc.)
AI_MODEL=deepseek/deepseek-chat-v3-0324:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Fallback models (comma-separated, tried in order on rate limit)
FALLBACK_MODELS=openai/gpt-3.5-turbo,anthropic/claude-3-haiku
```

---

## Phase 3: Visual Overhaul

### 3.1 Sidebar redesign

- Cleaner layout with better spacing
- Remove redundant status messages
- Group settings into collapsible sections using `st.expander()`:
  - AI Model Configuration
  - Analysis Context (location, industry)
  - Analysis Settings
- Add a compact "About" section at the bottom with version info (v2.1.0)

### 3.2 Header area

- Cleaner title: "AI Resume Analyzer"
- Subtitle: "Upload your resume and a job description to get AI-powered insights"
- Remove redundant emoji from title

### 3.3 Tab design

- Better tab content spacing
- Consistent padding inside tab panels

### 3.4 Cards and containers

- Refined card styling with subtle borders (1px solid border-color)
- Border-radius: 8px for cards, 6px for inputs
- Better padding (1rem-1.5rem) and margins (0.75rem-1rem)

### 3.5 Buttons

- Primary buttons: solid accent color, clean hover state
- Secondary buttons: outlined style
- Better sizing and spacing between buttons

### 3.6 Typography

- Better font sizing hierarchy (h1: 2rem, h2: 1.5rem, h3: 1.25rem)
- Improved line-height (1.6 for body text)
- Consistent heading styles

### 3.7 Minimal CSS stylesheet

Replace `static/styles.css` with brand polish only. The file should contain approximately 80-120 lines of CSS covering:
- `.stButton > button` — accent color, hover effect, border-radius
- `.stTabs [data-baseweb="tab"]` — active tab accent underline
- `[data-testid="stExpander"]` — subtle border and padding
- `[data-testid="stMetric"]` — card-like styling with border
- `.block-container` — max-width and padding adjustments
- NO background colors, text colors, or font-family overrides

---

## Phase 4: Error Handling

### 4.1 API error mapping

**File:** `utils/ai_analyzer.py`

In `_make_api_request()`, replace generic error messages with user-friendly ones:

| Status Code | Message |
|-------------|---------|
| 401 | "Invalid API key. Check your .env file." |
| 402 | "Insufficient credits. Add credits to your OpenRouter account." |
| 429 | "Rate limited. Retrying with fallback model..." |
| 500+ | "Service temporarily unavailable. Try again in a moment." |
| Timeout | "Request timed out. Check your internet connection." |
| Connection | "Unable to connect to API. Check your internet connection." |

### 4.2 Analysis progress indicators

**File:** `app.py`

Use Streamlit's `st.status()` context manager for multi-step progress:
```python
with st.status("🔄 Analyzing resume...", expanded=True) as status:
    st.write("📄 Extracting text from files...")
    # ... extract text
    st.write("🤖 Performing AI analysis...")
    # ... run analysis
    st.write("✨ Generating insights...")
    # ... compile results
    status.update(label="✅ Analysis complete!", state="complete")
```

If `st.status()` is not available (Streamlit < 1.28), fall back to `st.spinner()` with step text.

### 4.3 Partial failure resilience

**File:** `utils/ai_analyzer.py`

In `analyze_resume_comprehensive()`, wrap each analysis step in try/except:
- If salary estimation fails, continue with other analyses and note the failure
- If skill gap fails, continue and return empty skill gap
- Return partial results instead of failing completely

### 4.4 Resume rewrite error handling

**File:** `utils/resume_writer.py`

In `_rewrite_section()`:
- If a section fails to rewrite, return the original text with a warning
- Add a "Retry" button per section in the UI
- Show which sections failed and why

### 4.5 Troubleshooting section

**File:** `app.py`

Add a collapsible "Troubleshooting" section in the sidebar:
- "AI Analysis Not Working" → Check API key, credits, internet
- "PDF Extraction Fails" → Ensure PDF is not password-protected
- "Installation Issues" → Run pip install, check Python version

---

## Phase 5: Analysis Quality

### 5.1 Prompt improvements

**File:** `utils/ai_analyzer.py`

**Match score prompt:** Add scoring rubric:
```
Score each category 0-100:
- 90-100: Expert/Exceptional match
- 70-89: Strong/Proficient match
- 50-69: Moderate/Adequate match
- 30-49: Below average/Gaps present
- 0-29: Poor/Significant gaps
```

**Skill gap prompt:** Add time estimates:
```
For each skill gap, estimate development time:
- "1-2 weeks" for basic proficiency
- "1-3 months" for working knowledge
- "3-6 months" for competence
- "6+ months" for expertise
```

**ATS prompt:** Add placement suggestions:
```
For each missing keyword, suggest where to add it:
- "Add to Skills section"
- "Include in Experience bullet points"
- "Mention in Summary/Objective"
```

### 5.2 JSON parsing resilience

**File:** `utils/ai_analyzer.py`

Add a `_parse_json_response()` helper method:
1. Try `json.loads(response)` directly
2. If fails, try extracting JSON from markdown fences: ` ```json ... ``` `
3. If fails, try finding first `{` and last `}` and parsing that substring
4. If fails, try cleaning common issues (trailing commas, single quotes)
5. If all fails, return a fallback result with raw text in explanation field

### 5.3 Configuration flags

**File:** `utils/ai_analyzer.py`

Use existing env vars to control which analyses run:
```python
self.enable_salary = os.getenv('ENABLE_SALARY_ESTIMATION', 'true').lower() == 'true'
self.enable_skill_gap = os.getenv('ENABLE_SKILL_GAP_ANALYSIS', 'true').lower() == 'true'
self.enable_ats = os.getenv('ENABLE_ATS_OPTIMIZATION', 'true').lower() == 'true'
```

Skip disabled analyses in `analyze_resume_comprehensive()` and return None for their results.

---

## File Changes Summary

| File | Action |
|------|--------|
| `.streamlit/config.toml` | **NEW** — Streamlit theme configuration |
| `static/styles.css` | Replace — minimal brand polish only |
| `app.py` | Modify — remove theme system, add progress indicators, sidebar redesign, error handling |
| `utils/ai_analyzer.py` | Modify — rename env var, add JSON parsing, improve prompts, partial failure resilience |
| `utils/resume_writer.py` | Modify — better error handling per section |
| `.env.example` | Modify — rename DEEPSEEK_MODEL to AI_MODEL |
| `README.md` | Modify — update theme and model documentation |
