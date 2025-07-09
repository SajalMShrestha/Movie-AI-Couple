# 🎬 Complete Setup Instructions for Movie Recommendation System

## ✅ Step-by-Step Setup

### 1. Create Project Structure
```bash
cd Desktop/Movie_AI_3

# Create directories
mkdir src tests

# Create __init__.py files
touch src/__init__.py
touch tests/__init__.py
```

### 2. Copy All Files

Copy the provided code into these files:

**Source Files (src/ directory):**
- `src/utils.py` - Constants and utility functions
- `src/movie_search.py` - Search and fuzzy matching
- `src/narrative_analysis.py` - Text analysis and narrative style
- `src/franchise_detection.py` - Franchise grouping logic
- `src/feedback_system.py` - Google Sheets integration
- `src/movie_scoring.py` - Recommendation engine

**Main Application:**
- `main_app.py` - Streamlit application

**Test Files (tests/ directory):**
- `tests/test_movie_search.py` - Search functionality tests
- `tests/test_narrative_analysis.py` - Text analysis tests
- `tests/test_franchise_detection.py` - Franchise detection tests
- `tests/test_feedback_system.py` - Feedback system tests
- `tests/test_movie_scoring.py` - Scoring algorithm tests
- `tests/test_utils.py` - Utility function tests

**Additional Files:**
- `test_runner.py` - Comprehensive test runner
- `requirements.txt` - Updated dependencies

### 3. Keep Your Existing Files
✅ **Keep these exactly where they are:**
- `.streamlit/secrets.toml` - Your API keys and credentials
- `user_feedback.csv` - Your existing feedback data
- `session_map.csv` - Your session mapping

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run Tests
```bash
# Check dependencies
python test_runner.py --deps

# Run all tests
python test_runner.py

# Run specific module tests
python test_runner.py movie_search
python test_runner.py narrative_analysis
```

### 6. Run Your Application
```bash
streamlit run main_app.py
```

## 🧪 Testing Commands

### Basic Testing
```bash
# Run all tests with detailed output
python test_runner.py

# Run specific test files
python -m pytest tests/test_movie_search.py -v
python -m pytest tests/test_utils.py -v
```

### Advanced Testing
```bash
# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test classes
python -m pytest tests/test_movie_search.py::TestMovieSearch -v

# Run tests and stop on first failure
python -m pytest tests/ -x
```

## 📁 Final Project Structure

```
Movie_AI_3/
├── .streamlit/
│   └── secrets.toml              ✅ Keep existing
├── src/
│   ├── __init__.py              🆕 Create empty
│   ├── utils.py                 🆕 Copy provided code
│   ├── movie_search.py          🆕 Copy provided code
│   ├── narrative_analysis.py    🆕 Copy provided code
│   ├── franchise_detection.py   🆕 Copy provided code
│   ├── feedback_system.py       🆕 Copy provided code
│   └── movie_scoring.py         🆕 Copy provided code
├── tests/
│   ├── __init__.py              🆕 Create empty
│   ├── test_movie_search.py     🆕 Copy provided code
│   ├── test_narrative_analysis.py 🆕 Copy provided code
│   ├── test_franchise_detection.py 🆕 Copy provided code
│   ├── test_feedback_system.py  🆕 Copy provided code
│   ├── test_movie_scoring.py    🆕 Copy provided code
│   └── test_utils.py            🆕 Copy provided code
├── main_app.py                  🆕 Copy provided code
├── test_runner.py               🆕 Copy provided code
├── requirements.txt             🆕 Update with provided version
├── user_feedback.csv            ✅ Keep existing
└── session_map.csv              ✅ Keep existing
```

## 🎯 Key Benefits of This Structure

### ✅ **Modularity**
- Each component has a single responsibility
- Easy to modify individual features
- Clear separation of concerns

### ✅ **Testability** 
- Comprehensive unit tests for all components
- Easy to test individual functions
- Mock external dependencies

### ✅ **Maintainability**
- Easy to find and fix bugs
- Clear code organization
- Well-documented functions

### ✅ **Scalability**
- Simple to add new features
- Easy to extend recommendation algorithms
- Modular architecture supports growth

## 🔧 Troubleshooting

### Import Errors
```bash
# If you get import errors, make sure __init__.py files exist:
touch src/__init__.py
touch tests/__init__.py
```

### Test Failures
```bash
# Some tests may fail due to missing API keys or network issues
# This is expected in a test environment
# Focus on the code structure and logic tests
```

### NLTK Downloads
```bash
# If NLTK data is missing:
python -c "import nltk; nltk.download('all')"
```

## 🚀 Next Steps

1. **Run the tests** to verify everything works
2. **Run your app** to make sure it still functions
3. **Add new features** using the modular structure
4. **Write tests** for any new functionality you add
5. **Use version control** (git) to track changes

## 📚 Development Workflow

### Adding New Features
1. Write the function in the appropriate `src/` module
2. Write tests in the corresponding `tests/` file
3. Run tests to verify functionality
4. Update main app to use new feature

### Debugging Issues
1. Run specific module tests: `python test_runner.py module_name`
2. Use print statements or debugger in individual modules
3. Check test output for clues about what's failing

### Code Quality
```bash
# Format code (optional)
pip install black
black src/ tests/ main_app.py

# Check code style (optional)  
pip install flake8
flake8 src/ tests/ main_app.py
```

This modular structure will make your movie recommendation system much more maintainable and easier to extend! 🎬✨