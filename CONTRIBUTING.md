# Contributing to Chaos RNG

First off, thank you for considering contributing to Chaos RNG! It's people like you that make this project a great tool for the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/chaos-rng.git
   cd chaos-rng
   ```

3. **Set up the development environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install in development mode with all dependencies
   pip install -e .[all]

   # Install pre-commit hooks
   pre-commit install
   ```

4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Reporting Bugs

- **Check existing issues** first to avoid duplicates
- **Use the bug report template** when creating a new issue
- **Include detailed information**:
  - Your operating system and Python version
  - Exact error messages
  - Minimal code example that reproduces the issue
  - Expected vs actual behavior

### Suggesting Enhancements

- **Check existing issues** and discussions first
- **Use the feature request template**
- **Provide detailed rationale** for the enhancement
- **Consider backwards compatibility**

### Code Contributions

We welcome contributions in these areas:

1. **Bug fixes**
2. **New features** (please discuss in an issue first)
3. **Performance improvements**
4. **Documentation improvements**
5. **Test coverage improvements**
6. **Statistical validation enhancements**

## Pull Request Process

1. **Ensure all tests pass** locally:
   ```bash
   make test
   ```

2. **Check code quality**:
   ```bash
   make lint
   make format-check
   make type-check
   ```

3. **Update documentation** if needed:
   - Update docstrings for any new/modified functions
   - Update README if adding new features
   - Add examples for new functionality

4. **Add tests** for new functionality:
   - Unit tests for individual functions
   - Integration tests for complex features
   - Statistical tests for randomness properties

5. **Update CHANGELOG.md** with your changes

6. **Submit the pull request**:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what your changes do and why

7. **Address review feedback** promptly

### Review Process

- All submissions require review before merging
- Reviews focus on:
  - Code correctness and quality
  - Test coverage and quality
  - Documentation completeness
  - Performance implications
  - API design consistency

## Style Guidelines

### Code Style

We use automated code formatting and linting:

- **Black** for code formatting (line length: 88)
- **Ruff** for linting and import sorting
- **MyPy** for type checking

Run these tools locally:
```bash
make format     # Auto-format code
make lint       # Check for linting issues
make type-check # Run type checking
```

### Coding Standards

1. **Type hints**: All public functions must have type hints
2. **Docstrings**: All public functions/classes need comprehensive docstrings
3. **Error handling**: Use appropriate exceptions with clear messages
4. **Performance**: Consider NumPy vectorization for numerical code
5. **Thread safety**: Ensure new code maintains thread safety

### Docstring Format

Use NumPy-style docstrings:

```python
def calculate_lyapunov_exponent(
    positions: np.ndarray,
    dt: float,
    steps: int
) -> float:
    """
    Calculate the largest Lyapunov exponent for a chaotic system.

    Parameters
    ----------
    positions : np.ndarray
        Array of position vectors with shape (n_steps, n_bodies, 3).
    dt : float
        Time step size.
    steps : int
        Number of steps to analyze.

    Returns
    -------
    float
        The largest Lyapunov exponent.

    Raises
    ------
    ValueError
        If positions array has incorrect shape.

    Examples
    --------
    >>> positions = simulate_three_body(steps=1000)
    >>> lyapunov = calculate_lyapunov_exponent(positions, 0.01, 1000)
    >>> lyapunov > 0  # Should be positive for chaotic systems
    True
    """
```

## Testing

### Test Categories

1. **Unit tests**: Test individual functions in isolation
2. **Integration tests**: Test component interactions
3. **Statistical tests**: Verify randomness properties (NIST tests)
4. **Performance tests**: Benchmark critical functions
5. **Property-based tests**: Use Hypothesis for edge cases

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/ -m "not slow"           # Fast tests only
pytest tests/ -m statistical          # Statistical tests
pytest tests/ -m integration          # Integration tests

# Run with coverage
make test-coverage

# Run performance benchmarks
pytest tests/ -k benchmark --benchmark-only
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure: `tests/test_module.py` for `src/chaos_rng/module.py`
- Use descriptive test names: `test_three_body_generates_chaotic_dynamics`
- Include edge cases and error conditions
- Use fixtures for common test data

### Statistical Testing

For randomness-related code:

1. **Include appropriate statistical tests**
2. **Use sufficient sample sizes** for reliable results
3. **Set appropriate significance levels**
4. **Document expected behavior** clearly

## Documentation

### Building Documentation

```bash
# Build HTML documentation
make docs

# Build and serve locally
make docs-serve
```

### Documentation Guidelines

1. **Keep README up-to-date** with new features
2. **Document API changes** thoroughly
3. **Include practical examples**
4. **Update version compatibility** information
5. **Link to relevant research papers** for mathematical concepts

## Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Email**: For security-related concerns

## Recognition

Contributors will be recognized in:
- GitHub's contributor graph
- Release notes for significant contributions
- Optional AUTHORS file for major contributors

Thank you for contributing to Chaos RNG! 🎲
