# Contributing to OpenRubricRL

Thank you for your interest in contributing to OpenRubricRL! This guide will help you get started with contributing to our open-source pipeline for converting human-written rubrics into LLM-based reward functions.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inspiring community for all.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A GitHub account

### Areas for Contribution

We welcome contributions in several areas:

- **Core Features**: Rubric parsing, prompt generation, scoring algorithms
- **Integrations**: New RL libraries, LLM providers, evaluation frameworks
- **Documentation**: Tutorials, API docs, examples
- **Testing**: Unit tests, integration tests, performance tests
- **Bug Fixes**: Issue resolution and stability improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/anikal2001/OpenRubricRL.git
cd OpenRubricRL
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode with all dependencies
pip install -e ".[dev,all]"
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code formatting and linting checks before each commit.

### 4. Verify Installation

```bash
# Run tests to ensure everything is working
pytest

# Try the CLI
openrubricrl --help
```

## Making Changes

### Branch Strategy

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number
   ```

2. Make your changes in logical, atomic commits
3. Write clear commit messages following conventional commits format:
   ```
   feat: add support for custom scoring metrics
   fix: resolve rubric validation edge case
   docs: update API documentation for scoring endpoint
   test: add integration tests for anthropic provider
   ```

### Project Structure

```
OpenRubricRL/
├── src/openrubricrl/          # Main package source
│   ├── api/                   # FastAPI endpoints
│   ├── core/                  # Core rubric and scoring logic
│   ├── integrations/          # RL library integrations
│   ├── logging/               # Logging utilities
│   └── cli.py                 # Command-line interface
├── tests/                     # Test suite
├── examples/                  # Usage examples
├── docs/                      # Documentation
└── pyproject.toml            # Project configuration
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=openrubricrl

# Run specific test file
pytest tests/test_rubric.py

# Run tests matching a pattern
pytest -k "test_scoring"
```

### Writing Tests

- Write tests for all new functionality
- Aim for high test coverage (>90%)
- Use descriptive test names that explain what is being tested
- Include both unit tests and integration tests where appropriate

Example test structure:
```python
def test_rubric_validation_with_valid_schema():
    """Test that valid rubric schemas pass validation."""
    # Arrange
    valid_rubric = {...}
    
    # Act
    result = validate_rubric(valid_rubric)
    
    # Assert
    assert result.is_valid
    assert len(result.errors) == 0
```

## Code Style

We use several tools to maintain code quality:

### Formatting and Linting

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run these manually:
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use descriptive variable and function names

Example function with proper style:
```python
def calculate_weighted_score(
    criteria_scores: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """Calculate weighted average score from criteria scores.
    
    Args:
        criteria_scores: Mapping of criteria names to their scores
        weights: Mapping of criteria names to their weights
        
    Returns:
        Weighted average score between 0.0 and 1.0
        
    Raises:
        ValueError: If criteria_scores and weights have mismatched keys
    """
    if set(criteria_scores.keys()) != set(weights.keys()):
        raise ValueError("Criteria scores and weights must have matching keys")
    
    total_weighted = sum(
        score * weights[criterion] 
        for criterion, score in criteria_scores.items()
    )
    total_weight = sum(weights.values())
    
    return total_weighted / total_weight if total_weight > 0 else 0.0
```

## Submitting Changes

### Before Submitting

1. **Run the full test suite**: `pytest`
2. **Check code style**: `pre-commit run --all-files`
3. **Update documentation** if you've changed APIs
4. **Add tests** for new functionality
5. **Update CHANGELOG.md** if applicable

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the changes and motivation

## Issue Guidelines

### Reporting Bugs

When reporting bugs, please include:

- **Environment details**: Python version, OS, OpenRubricRL version
- **Steps to reproduce**: Minimal example that demonstrates the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Full traceback if applicable

### Feature Requests

For feature requests, please provide:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Additional context**: Any other relevant information

### Issue Labels

We use labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

## Pull Request Process

1. **Create an issue** first to discuss major changes
2. **Fork the repository** and create a feature branch
3. **Make your changes** following the guidelines above
4. **Test thoroughly** and ensure all checks pass
5. **Submit a pull request** with a clear description

### PR Review Process

- All PRs require at least one review from a maintainer
- Automated checks must pass (tests, linting, type checking)
- Large changes may require additional review
- We aim to review PRs within 48 hours

### After Your PR is Merged

- Delete your feature branch
- Pull the latest changes from main
- Consider contributing to related issues or documentation

## Community

### Getting Help

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs and request features
- **Documentation**: Documentation is available in the [docs](https://github.com/anikal2001/OpenRubricRL/tree/main/docs) directory

### Recognition

Contributors are recognized in:
- The project README
- Release notes for significant contributions
- Our contributor hall of fame

## Development Tips

### Useful Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run specific test categories
pytest tests/test_api.py -v
pytest tests/ -k "integration"

# Generate test coverage report
pytest --cov=openrubricrl --cov-report=html

# Check type hints
mypy src/openrubricrl

# Format and lint
black . && isort . && flake8
```

### Debugging

- Use the `--debug` flag with CLI commands for verbose output
- Set `OPENRUBRICRL_LOG_LEVEL=DEBUG` for detailed logging
- Use `pytest -s` to see print statements during tests

### Working with APIs

When testing API changes:
```bash
# Start the development server
uvicorn openrubricrl.api.main:app --reload

# Test endpoints
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{"rubric": {...}, "content": "..."}'
```

---

Thank you for contributing to OpenRubricRL! Your efforts help make RLHF more accessible and effective for everyone. 🚀