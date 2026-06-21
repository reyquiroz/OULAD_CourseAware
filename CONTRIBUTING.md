# Contributing to OULAD At-Risk Student Prediction

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Prioritize the educational mission of the project

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/OULAD.git
   cd OULAD
   ```
3. **Set up the development environment** (see below)
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Found a bug? Open an issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

2. **Feature Requests**: Have an idea? Open an issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Any alternatives you've considered

3. **Code Contributions**:
   - Bug fixes
   - New features
   - Performance improvements
   - Documentation improvements

4. **Documentation**:
   - Fix typos or clarify existing docs
   - Add examples or tutorials
   - Improve code comments

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Create virtual environment**:
   ```bash
   python -m venv oulad_env
   source oulad_env/bin/activate  # On Windows: oulad_env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download OULAD dataset**:
   - Visit: https://analyse.kmi.open.ac.uk/open_dataset
   - Download all CSV files
   - Place in `data/raw/` directory

4. **Verify setup**:
   ```bash
   python src/baseline_evaluation.py
   ```

## Coding Standards

### Python Style Guide
- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Code Organization
```python
# Good example
def calculate_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Area Under ROC Curve.
    
    Args:
        y_true: True labels (1=at-risk, 0=success)
        y_pred: Predicted probabilities
        
    Returns:
        AUROC score between 0 and 1
    """
    return roc_auc_score(y_true, y_pred)
```

### Label Convention (CRITICAL)
**Always use the correct label convention:**
- **1 = at-risk** (Fail/Withdrawn) - students requiring intervention
- **0 = success** (Pass/Distinction) - students on track

All metrics (precision, recall, F1, AUPRC) refer to identifying at-risk students.

### File Organization
- **Source code**: `src/`
- **Notebooks**: `notebooks/`
- **Documentation**: `docs/`
- **Results**: `results/`
- **Tests**: `tests/` (when added)

## Testing

### Running Tests
```bash
# When tests are added
pytest tests/
```

### Writing Tests
- Write unit tests for new functions
- Include edge cases
- Test with different data splits
- Verify temporal leakage prevention

### Manual Testing Checklist
Before submitting a PR, verify:
- [ ] Code runs without errors
- [ ] Results are reproducible
- [ ] No temporal leakage (future data not used)
- [ ] Label convention is correct
- [ ] Documentation is updated
- [ ] All paths are relative

## Documentation

### Code Documentation
- Add docstrings to all functions and classes
- Use Google-style docstrings
- Include examples where helpful

### README Updates
- Update README.md if adding new features
- Add usage examples
- Update installation instructions if needed

### Documentation Files
- Update relevant docs in `docs/` directory
- Keep QUICK_START.md current
- Update EXECUTION_GUIDE.md for new scripts

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run checks**:
   - Code runs without errors
   - Tests pass (when available)
   - Documentation is updated
   - No merge conflicts

3. **Commit messages**:
   - Use clear, descriptive commit messages
   - Reference issues: "Fix #123: Description"
   - Format: `type: description`
     - `feat:` new feature
     - `fix:` bug fix
     - `docs:` documentation
     - `refactor:` code refactoring
     - `test:` adding tests

### Submitting PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to related issues
   - Screenshots (if UI changes)
   - Test results

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Tested locally
   - [ ] All checks pass
   - [ ] Documentation updated
   
   ## Related Issues
   Fixes #123
   ```

### Review Process

1. **Automated checks** will run (when CI/CD is set up)
2. **Maintainer review**: Address feedback promptly
3. **Approval**: Once approved, maintainer will merge
4. **Celebrate**: Your contribution is now part of the project! 🎉

## Project-Specific Guidelines

### Temporal Leakage Prevention
When adding features or modifying data processing:
- **Always filter by prediction window**
- **Never use future information**
- **Document temporal assumptions**
- **Test with different prediction windows**

### Cross-Course Evaluation
When modifying evaluation code:
- **Maintain LCPO split integrity**
- **Test on multiple courses**
- **Document course-specific behavior**
- **Report performance variation**

### Feature Engineering
When adding new features:
- **Document feature meaning**
- **Specify temporal availability**
- **Test for leakage**
- **Analyze feature importance**

## Questions?

- **Open an issue** for questions about contributing
- **Check existing issues** for similar questions
- **Read the documentation** in `docs/` directory
- **Review QUICK_START.md** for basic usage

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Project documentation
- Future publications (for significant contributions)

Thank you for contributing to educational data mining and helping improve student success prediction!