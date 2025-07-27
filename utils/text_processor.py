"""
Text Processor Module
====================

Handles text preprocessing, cleaning, and NLP operations using spaCy.
Includes tokenization, lemmatization, stop word removal, and keyword extraction.
"""

import re
import string
import logging
from typing import List, Set, Tuple, Optional, Dict, Any
from collections import Counter

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    STOP_WORDS = set()

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Advanced text processor that handles cleaning, tokenization, and keyword extraction.
    
    Uses spaCy as the primary NLP engine with NLTK as fallback.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm", use_spacy: bool = True):
        """
        Initialize the TextProcessor.
        
        Args:
            model_name (str): spaCy model name to load
            use_spacy (bool): Whether to use spaCy (True) or NLTK (False)
        """
        self.model_name = model_name
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.nlp = None
        self.lemmatizer = None
        
        # Initialize NLP engine
        self._initialize_nlp()
        
        # Define comprehensive IT/CS skill-related keywords that should be preserved
        self.skill_keywords = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'go', 'rust',
                'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'dart',
                'objective-c', 'perl', 'shell', 'bash', 'powershell', 'vb.net', 'cobol',
                'assembly', 'lua', 'groovy', 'erlang', 'haskell', 'clojure'
            ],
            'web_frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring', 'laravel',
                'express', 'node.js', 'next.js', 'nuxt.js', 'svelte', 'ember', 'backbone',
                'jquery', 'bootstrap', 'tailwind', 'material-ui', 'chakra-ui'
            ],
            'mobile_frameworks': [
                'flutter', 'react-native', 'xamarin', 'ionic', 'cordova', 'phonegap',
                'native-script', 'unity', 'unreal-engine'
            ],
            'ml_ai_frameworks': [
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
                'opencv', 'yolo', 'transformers', 'huggingface', 'spacy', 'nltk',
                'matplotlib', 'seaborn', 'plotly', 'jupyter', 'anaconda'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
                'sql-server', 'cassandra', 'dynamodb', 'firebase', 'firestore',
                'elasticsearch', 'neo4j', 'couchdb', 'mariadb'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'google-cloud', 'heroku', 'digitalocean',
                'linode', 'vultr', 'cloudflare', 'netlify', 'vercel'
            ],
            'devops_tools': [
                'docker', 'kubernetes', 'jenkins', 'gitlab-ci', 'github-actions',
                'travis-ci', 'circleci', 'ansible', 'terraform', 'vagrant',
                'nginx', 'apache', 'linux', 'ubuntu', 'centos'
            ],
            'version_control': [
                'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial'
            ],
            'development_tools': [
                'vscode', 'intellij', 'eclipse', 'android-studio', 'xcode',
                'visual-studio', 'sublime', 'atom', 'vim', 'emacs', 'postman',
                'insomnia', 'swagger', 'figma', 'sketch'
            ],
            'web_technologies': [
                'html', 'css', 'sass', 'scss', 'less', 'webpack', 'babel',
                'typescript', 'json', 'xml', 'rest', 'graphql', 'soap',
                'ajax', 'cors', 'oauth', 'jwt', 'websockets'
            ],
            'testing_frameworks': [
                'jest', 'mocha', 'chai', 'cypress', 'selenium', 'pytest',
                'junit', 'testng', 'karma', 'jasmine', 'enzyme'
            ],
            'methodologies': [
                'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'ci/cd',
                'tdd', 'bdd', 'mvc', 'mvvm', 'microservices', 'api-first'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem-solving',
                'analytical', 'creative', 'adaptable', 'organized', 'collaboration',
                'mentoring', 'documentation', 'presentation', 'critical-thinking'
            ],
            'job_roles': [
                'developer', 'engineer', 'programmer', 'analyst', 'architect',
                'manager', 'lead', 'senior', 'junior', 'intern', 'graduate',
                'full-stack', 'front-end', 'back-end', 'mobile', 'web', 'data',
                'machine-learning', 'ai', 'devops', 'qa', 'tester'
            ],
            'cs_concepts': [
                'algorithm', 'data-structure', 'object-oriented', 'functional',
                'design-pattern', 'software-architecture', 'system-design',
                'database-design', 'api-design', 'security', 'performance',
                'scalability', 'optimization', 'debugging', 'refactoring'
            ]
        }
        
        # Flatten skill keywords for easy lookup
        self.all_skills = set()
        for category in self.skill_keywords.values():
            self.all_skills.update(category)
    
    def _initialize_nlp(self):
        """Initialize the NLP engine (spaCy or NLTK)."""
        if self.use_spacy:
            try:
                self.nlp = spacy.load(self.model_name)
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning(f"spaCy model '{self.model_name}' not found. Install with: python -m spacy download {self.model_name}")
                self.use_spacy = False
                self._initialize_nltk()
        else:
            self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Initialize NLTK as fallback."""
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                
                self.lemmatizer = WordNetLemmatizer()
                logger.info("NLTK initialized successfully")
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
        else:
            logger.warning("Neither spaCy nor NLTK available. Limited text processing capabilities.")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and common punctuation
        text = re.sub(r'[^\w\s\-\+\#\.]', ' ', text)
        
        # Remove standalone numbers (but keep version numbers like "3.8")
        text = re.sub(r'\b\d+\b', ' ', text)
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text.
        
        Args:
            text (str): Text to process
            
        Returns:
            List[str]: List of processed tokens
        """
        if not text:
            return []
        
        if self.use_spacy and self.nlp:
            return self._process_with_spacy(text)
        elif NLTK_AVAILABLE:
            return self._process_with_nltk(text)
        else:
            # Basic fallback processing
            return self._basic_tokenize(text)
    
    def _process_with_spacy(self, text: str) -> List[str]:
        """Process text using spaCy."""
        doc = self.nlp(text)
        tokens = []
        
        for token in doc:
            # Skip stop words, punctuation, and spaces
            if (token.is_stop or token.is_punct or token.is_space or 
                len(token.text) < 2):
                continue
            
            # Keep important skills even if they're marked as stop words
            if token.lemma_.lower() in self.all_skills:
                tokens.append(token.lemma_.lower())
            elif not token.is_stop:
                tokens.append(token.lemma_.lower())
        
        return tokens
    
    def _process_with_nltk(self, text: str) -> List[str]:
        """Process text using NLTK."""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Get English stop words
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            stop_words = set()
        
        processed_tokens = []
        
        for token in tokens:
            # Skip short tokens, punctuation, and stop words
            if (len(token) < 2 or token in string.punctuation or 
                token.lower() in stop_words):
                continue
            
            # Keep important skills
            if token.lower() in self.all_skills:
                processed_tokens.append(token.lower())
            elif token.lower() not in stop_words:
                # Lemmatize
                if self.lemmatizer:
                    lemmatized = self.lemmatizer.lemmatize(token.lower())
                    processed_tokens.append(lemmatized)
                else:
                    processed_tokens.append(token.lower())
        
        return processed_tokens
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization without NLP libraries."""
        # Simple word splitting and cleaning
        tokens = text.split()
        processed = []
        
        basic_stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        for token in tokens:
            # Clean token
            token = re.sub(r'[^\w]', '', token).lower()
            
            # Skip short tokens and basic stop words
            if len(token) < 2 or token in basic_stop_words:
                continue
            
            processed.append(token)
        
        return processed
    
    def _normalize_tech_terms(self, text: str) -> str:
        """
        Normalize common IT/CS terms and their variations.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Text with normalized tech terms
        """
        # Define synonym mappings for IT/CS terms
        tech_synonyms = {
            # Programming languages variations
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'c++': 'cpp',
            'c#': 'csharp',
            '.net': 'dotnet',
            'node': 'nodejs',
            'react.js': 'react',
            'vue.js': 'vue',
            'angular.js': 'angular',
            
            # Framework variations
            'django': 'django',
            'tensorflow': 'tensorflow',
            'tf': 'tensorflow',
            'pytorch': 'pytorch',
            'scikit-learn': 'sklearn',
            'pandas': 'pandas',
            'numpy': 'numpy',
            
            # Database variations
            'postgres': 'postgresql',
            'mongo': 'mongodb',
            'mysql': 'mysql',
            'sqlite': 'sqlite',
            
            # Cloud variations
            'amazon web services': 'aws',
            'google cloud platform': 'gcp',
            'google cloud': 'gcp',
            'microsoft azure': 'azure',
            
            # Mobile development
            'android development': 'android',
            'ios development': 'ios',
            'mobile development': 'mobile',
            'app development': 'mobile',
            'cross-platform': 'cross-platform',
            
            # Web development
            'web development': 'web',
            'frontend': 'front-end',
            'backend': 'back-end',
            'fullstack': 'full-stack',
            'full stack': 'full-stack',
            
            # Common tech terms
            'artificial intelligence': 'ai',
            'machine learning': 'ml',
            'deep learning': 'dl',
            'natural language processing': 'nlp',
            'computer vision': 'cv',
            'data science': 'data-science',
            'software engineering': 'software-engineering',
            'computer science': 'computer-science',
            
            # Version control
            'version control': 'git',
            'source control': 'git',
            
            # Development practices
            'continuous integration': 'ci',
            'continuous deployment': 'cd',
            'test driven development': 'tdd',
            'behavior driven development': 'bdd',
        }
        
        # Apply normalizations
        normalized_text = text.lower()
        for synonym, standard in tech_synonyms.items():
            normalized_text = normalized_text.replace(synonym, standard)
        
        return normalized_text
    
    def extract_keywords(self, text: str, top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Extract top keywords from text with IT/CS term normalization.
        
        Args:
            text (str): Text to extract keywords from
            top_k (int): Number of top keywords to return
            
        Returns:
            List[Tuple[str, int]]: List of (keyword, frequency) tuples
        """
        # Normalize tech terms first
        normalized_text = self._normalize_tech_terms(text)
        
        # Process text
        tokens = self.tokenize_and_lemmatize(normalized_text)
        
        if not tokens:
            return []
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Boost important tech terms
        for token in token_counts:
            if token in self.all_skills:
                token_counts[token] *= 2  # Double weight for recognized skills
        
        # Get top keywords
        top_keywords = token_counts.most_common(top_k)
        
        return top_keywords
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills by category from text with IT/CS term normalization.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, List[str]]: Skills grouped by category
        """
        # Normalize tech terms first
        normalized_text = self._normalize_tech_terms(text)
        
        # Process text
        tokens = self.tokenize_and_lemmatize(normalized_text)
        text_lower = normalized_text.lower()
        
        found_skills = {category: [] for category in self.skill_keywords.keys()}
        
        # Check for exact skill matches
        for category, skills in self.skill_keywords.items():
            for skill in skills:
                # Check both in tokens and raw text (for multi-word skills)
                if skill in tokens or skill in text_lower:
                    if skill not in found_skills[category]:
                        found_skills[category].append(skill)
        
        # Remove empty categories
        found_skills = {k: v for k, v in found_skills.items() if v}
        
        return found_skills
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get comprehensive text statistics.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, Any]: Text statistics
        """
        if not text:
            return {}
        
        # Basic stats
        words = text.split()
        sentences = text.split('.')
        
        # Process with NLP
        tokens = self.tokenize_and_lemmatize(text)
        keywords = self.extract_keywords(text, top_k=10)
        skills = self.extract_skills(text)
        
        stats = {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'unique_tokens': len(set(tokens)),
            'total_tokens': len(tokens),
            'top_keywords': keywords,
            'skills_found': skills,
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }
        
        return stats
    
    def is_nlp_available(self) -> Dict[str, bool]:
        """Check which NLP libraries are available."""
        return {
            'spacy': SPACY_AVAILABLE and self.nlp is not None,
            'nltk': NLTK_AVAILABLE,
            'current_engine': 'spacy' if self.use_spacy and self.nlp else 'nltk' if NLTK_AVAILABLE else 'basic'
        }
