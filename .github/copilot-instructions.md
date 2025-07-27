<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# AI Resume Analyzer - Copilot Instructions

## Project Overview
This is an AI Resume Analyzer built with Python that compares resumes with job descriptions using NLP techniques.

## Key Technologies
- **NLP**: spaCy for text processing, tokenization, and named entity recognition
- **ML**: scikit-learn for TF-IDF vectorization and cosine similarity
- **PDF Processing**: PyMuPDF for extracting text from PDF files
- **Web Interface**: Streamlit for interactive dashboard
- **Visualization**: matplotlib, seaborn, plotly for charts and graphs

## Code Style Guidelines
- Follow PEP 8 standards
- Use type hints for function parameters and return values
- Include comprehensive docstrings for all functions and classes
- Handle exceptions gracefully with informative error messages
- Use meaningful variable names that describe the data they hold

## Architecture Principles
- Modular design with separate utilities for different functions
- Clean separation between data processing and presentation
- Error handling and input validation throughout
- Efficient text processing with minimal memory usage
- Scalable design for future enhancements

## Testing Guidelines
- Write unit tests for all utility functions
- Include edge case testing for file processing
- Test with various resume and job description formats
- Validate similarity scoring accuracy with known test cases

When generating code, prioritize:
1. Readable, maintainable code structure
2. Proper error handling and user feedback
3. Efficient NLP processing techniques
4. Clear documentation and comments
5. Modular, reusable components
