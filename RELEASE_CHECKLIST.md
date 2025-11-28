# Release Checklist

This checklist ensures that all steps are completed before publishing a new version of Chaos RNG.

## Pre-Release Preparation

### Code Quality
- [ ] All tests pass locally (`make test`)
- [ ] All tests pass on CI (check GitHub Actions)
- [ ] Code quality checks pass (`make lint`)
- [ ] Type checking passes (`make type-check`)
- [ ] Security scans pass (`make security`)
- [ ] No critical security vulnerabilities in dependencies
- [ ] Pre-commit hooks pass (`make pre-commit`)

### Documentation
- [ ] README.md is up to date
- [ ] CHANGELOG.md is updated with new version changes
- [ ] API documentation is current
- [ ] Documentation builds without errors (`make docs`)
- [ ] All links in documentation work (`make docs-linkcheck`)
- [ ] Examples in documentation are tested and working

### Version Management
- [ ] Version number follows semantic versioning
- [ ] Version is bumped in all necessary files:
  - [ ] `pyproject.toml`
  - [ ] `src/chaos_rng/__init__.py`
  - [ ] `docs/conf.py`
- [ ] Git tag will be created with correct format (`v{version}`)

### Testing
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Statistical tests pass (NIST suite)
- [ ] Performance benchmarks run successfully
- [ ] Cross-platform testing completed (Linux, Windows, macOS)
- [ ] Multi-Python version testing completed (3.9-3.12)

## Release Process

### 1. Final Code Review
- [ ] All changes have been peer-reviewed
- [ ] No debug code or temporary changes remain
- [ ] All TODO comments are addressed or documented
- [ ] Code follows project style guidelines

### 2. Version Bump
Choose one:
- [ ] Patch version (bug fixes): `make bump-patch`
- [ ] Minor version (new features): `make bump-minor`
- [ ] Major version (breaking changes): `make bump-major`

### 3. Build and Test Package
- [ ] Clean build environment (`make clean`)
- [ ] Build package (`make build`)
- [ ] Check package contents (`make check-build`)
- [ ] Test installation in clean environment
- [ ] Verify package metadata is correct

### 4. Test PyPI Upload
- [ ] Upload to Test PyPI (`make upload-test`)
- [ ] Install from Test PyPI: `pip install -i https://test.pypi.org/simple/ chaos-rng=={version}`
- [ ] Test basic functionality
- [ ] Test all import paths
- [ ] Verify dependencies are correctly specified

### 5. GitHub Release
- [ ] Git tag is pushed: `git push origin main --tags`
- [ ] GitHub Actions release workflow starts automatically
- [ ] All release jobs complete successfully
- [ ] GitHub release is created with:
  - [ ] Correct version tag
  - [ ] Release notes from CHANGELOG
  - [ ] Built packages attached

### 6. PyPI Upload
- [ ] Automatic PyPI upload completes (via GitHub Actions)
- [ ] Package appears on PyPI: https://pypi.org/project/chaos-rng/
- [ ] Installation works: `pip install chaos-rng=={version}`
- [ ] All optional dependencies work: `pip install chaos-rng[crypto,test,dev,docs]`

## Post-Release Verification

### Package Verification
- [ ] Package installs correctly via pip
- [ ] All imports work correctly
- [ ] Basic functionality works
- [ ] Optional dependencies install correctly
- [ ] Documentation links work

### Distribution Verification
- [ ] PyPI page displays correctly
- [ ] GitHub release page displays correctly
- [ ] Documentation site updates (if applicable)
- [ ] Badge URLs in README work correctly

### Communication
- [ ] Release announcement prepared (if applicable)
- [ ] Dependencies that rely on this package are notified (if applicable)
- [ ] Community channels updated (if applicable)

## Rollback Plan

If issues are discovered after release:

### Minor Issues
- [ ] Document known issues
- [ ] Prepare patch release
- [ ] Follow same release process

### Critical Issues
- [ ] Contact PyPI to yank problematic release
- [ ] Document issues prominently
- [ ] Prepare emergency patch release
- [ ] Notify users through all available channels

## Environment Setup Commands

For release manager, ensure you have:

```bash
# Install release tools
pip install bump2version twine build

# Set up PyPI credentials (one time only)
# Create ~/.pypirc with your API tokens

# Set up GitHub CLI (optional)
gh auth login
```

## API Token Requirements

Ensure these secrets are configured in GitHub repository:
- [ ] `PYPI_API_TOKEN` - PyPI API token for publishing
- [ ] `TEST_PYPI_API_TOKEN` - Test PyPI API token for testing

## Version History

| Version | Release Date | Type | Notes |
|---------|-------------|------|-------|
| 0.1.0   | TBD         | Initial | First public release |

## Post-Release Tasks

After successful release:
- [ ] Update development dependencies
- [ ] Start new section in CHANGELOG.md for next version
- [ ] Update project roadmap (if applicable)
- [ ] Monitor for issues and user feedback

---

**Release Manager**: _______________
**Release Date**: _______________
**Version**: _______________
**Final Sign-off**: _______________
