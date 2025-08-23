# Chaos RNG - Complete Deployment Guide

This guide provides step-by-step instructions to transform your local chaos-rng project into a professionally published Python library.

## 🎯 Overview

Your project is now fully configured with professional-grade tooling:
- ✅ Complete GitHub Actions CI/CD pipeline
- ✅ Automated code quality and security checks
- ✅ Professional documentation setup
- ✅ PyPI publishing automation
- ✅ Development environment standardization

## 📋 Prerequisites

Before starting, ensure you have:
- Python 3.9+ installed
- Git installed and configured
- GitHub account
- PyPI account (create at https://pypi.org)
- Test PyPI account (create at https://test.pypi.org)

## 🚀 Step-by-Step Setup

### 1. Create GitHub Repository

```bash
# Navigate to your project
cd /home/ubuntumt/chaos-rng

# Initialize git (if not already done)
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial release preparation for Chaos RNG v0.1.0

- Complete project structure with professional tooling
- GitHub Actions CI/CD pipeline for testing and publishing
- Comprehensive documentation and contribution guidelines
- Security configurations and dependency management
- Development environment standardization"
```

**GitHub Repository Setup:**
1. Go to https://github.com/new
2. Create repository named `chaos-rng`
3. Make it public
4. Don't initialize with README (you already have one)

```bash
# Connect to GitHub
git remote add origin https://github.com/mustafatarim/chaos-rng.git
git branch -M main
git push -u origin main
```

### 2. Configure GitHub Repository Settings

**Branch Protection (GitHub web interface):**
1. Go to Settings → Branches
2. Add branch protection rule for `main`:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Include administrators

**GitHub Secrets (for PyPI publishing):**
1. Go to Settings → Secrets and variables → Actions
2. Add these repository secrets:
   - `PYPI_API_TOKEN`: Your PyPI API token
   - `TEST_PYPI_API_TOKEN`: Your Test PyPI API token

**How to get PyPI tokens:**
```bash
# 1. Create accounts at:
#    - https://pypi.org (production)
#    - https://test.pypi.org (testing)
# 
# 2. Generate API tokens:
#    - Go to Account Settings → API tokens
#    - Create token with scope for "Entire account"
#    - Copy the token (starts with pypi-...)
```

### 3. Set Up Development Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
make setup-dev

# Verify everything works
make test
make lint
make docs
```

### 4. Verify Project Configuration

Check that your pyproject.toml has correct information:
```bash
# Edit if needed:
# - author name and email
# - repository URLs
# - description
nano pyproject.toml
```

### 5. Test Local Build

```bash
# Clean and build
make clean
make build

# Check package
make check-build

# Test local installation
pip install dist/chaos_rng-0.1.0-py3-none-any.whl
python -c "import chaos_rng; print(chaos_rng.__version__)"
```

## 🧪 Testing Before Publication

### 1. Test with Test PyPI

```bash
# Upload to Test PyPI
make upload-test

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ chaos-rng==0.1.0

# Test functionality
python -c "
from chaos_rng import ThreeBodyRNG
rng = ThreeBodyRNG(seed=42)
print('Random numbers:', rng.random(5))
print('Test successful!')
"
```

### 2. Run Complete Test Suite

```bash
# Run all tests
make test

# Run statistical tests
make test-statistical

# Run security checks
make security

# Run full CI simulation
make ci
```

## 📦 Publishing to PyPI

### Option A: Automatic Release (Recommended)

```bash
# Create a release tag (this triggers automatic PyPI publishing)
git tag v0.1.0
git push origin v0.1.0

# OR use the Makefile for a complete release process
make release-minor  # Will test, build, tag, and push
```

### Option B: Manual Release

```bash
# Build and upload manually
make build
make upload
```

## 🔧 Post-Publication Setup

### 1. Enable GitHub Pages (for documentation)

1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: gh-pages (will be created by the docs workflow)

### 2. Configure ReadTheDocs (optional)

1. Go to https://readthedocs.org
2. Import your GitHub repository
3. Documentation will be available at https://chaos-rng.readthedocs.io

### 3. Add Badges to README

Your README already has badge placeholders. Update them with your actual repository:

```markdown
[![PyPI version](https://badge.fury.io/py/chaos-rng.svg)](https://badge.fury.io/py/chaos-rng)
[![Tests](https://github.com/mustafatarim/chaos-rng/workflows/Tests/badge.svg)](https://github.com/mustafatarim/chaos-rng/actions)
[![Documentation](https://readthedocs.org/projects/chaos-rng/badge/?version=latest)](https://chaos-rng.readthedocs.io)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/chaos-rng/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/chaos-rng)
```

## 🎛️ Development Workflow

### Daily Development

```bash
# Start development
make setup-dev

# Make changes, then:
make dev-check  # Quick quality check

# Before committing:
make qa  # Full quality assurance

# Commit with pre-commit hooks
git add .
git commit -m "Your changes"
git push
```

### Creating Releases

```bash
# For bug fixes
make release-patch

# For new features
make release-minor

# For breaking changes
make release-major
```

### Monitoring and Maintenance

```bash
# Update dependencies (run weekly)
pre-commit autoupdate
pip-review --local --auto  # If you have pip-review installed

# Check security
make security

# Run full test suite
make ci
```

## 📊 Package Monitoring

### PyPI Statistics
- View downloads: https://pypistats.org/packages/chaos-rng
- Package page: https://pypi.org/project/chaos-rng/

### GitHub Insights
- Repository insights for contributor activity
- Dependabot alerts for security updates
- Actions tab for CI/CD status

### Security Monitoring
- GitHub Security tab for vulnerability alerts
- Dependabot PRs for dependency updates
- CodeQL analysis results

## 🔍 Troubleshooting

### Common Issues

**Build Failures:**
```bash
# Clean everything and retry
make clean
rm -rf venv
python -m venv venv
source venv/bin/activate
make setup-dev
make build
```

**Test Failures:**
```bash
# Run specific test categories
make test-fast  # Skip slow statistical tests
make test-statistical  # Only statistical tests
```

**PyPI Upload Issues:**
```bash
# Check authentication
twine check dist/*

# Verify package contents
tar -tzf dist/chaos-rng-0.1.0.tar.gz | head -20
```

### Getting Help

- GitHub Issues: For bug reports and feature requests
- GitHub Discussions: For questions and community support
- Security Issues: mail@mustafatarim.com

## 📈 Version Management

This project uses semantic versioning (MAJOR.MINOR.PATCH):

- **PATCH** (0.1.1): Bug fixes, no new features
- **MINOR** (0.2.0): New features, backward compatible
- **MAJOR** (1.0.0): Breaking changes

Use the Makefile commands for automatic version management:
```bash
make bump-patch   # 0.1.0 → 0.1.1
make bump-minor   # 0.1.0 → 0.2.0
make bump-major   # 0.1.0 → 1.0.0
```

## ✅ Final Verification Checklist

Before your first release, verify:

- [ ] All tests pass: `make test`
- [ ] Code quality passes: `make lint`
- [ ] Documentation builds: `make docs`
- [ ] Security checks pass: `make security`
- [ ] Package builds correctly: `make build`
- [ ] Test PyPI upload works: `make upload-test`
- [ ] GitHub Actions workflows are green
- [ ] Repository settings are configured
- [ ] PyPI tokens are set as GitHub secrets

## 🎉 Success!

Once published, your package will be available via:

```bash
pip install chaos-rng
```

And your documentation will be available at your configured URL.

Welcome to the Python packaging ecosystem! 🚀

---

**Next Steps:**
1. Follow the steps in order
2. Test everything thoroughly
3. Publish with confidence
4. Monitor and maintain regularly
