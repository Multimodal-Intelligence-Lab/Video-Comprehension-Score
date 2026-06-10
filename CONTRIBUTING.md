# Contributing Guide

Thank you for your interest in contributing to VCS Metrics! This guide will help you get started with the development workflow.

## 🚀 Quick Start

### 1. Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/Multimodal-Intelligence-Lab/Video-Comprehension-Score.git
cd Video-Comprehension-Score

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies (torch installs automatically as of 2.0.0;
# pre-install a specific CUDA/CPU build first if you need one)
pip install -e .[dev,test]
```

### 2. Make Your Changes
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes to src/vcs/
# Add tests if needed
# Update documentation if necessary
```

### 3. Commit Your Changes
Use clear, descriptive commit messages:

```bash
# Use standard descriptive commit messages
git commit -m "add support for custom thresholds"
git commit -m "redesign API for better performance"
git commit -m "fix calculation error in similarity metric"
git commit -m "improve code documentation"
git commit -m "update README with examples"

# Version numbers are manually managed during releases
# No special commit message format required
```

### 4. Submit Pull Request
```bash
# Push your branch
git push origin feature/your-feature-name

# Create Pull Request via GitHub UI
# → Automatic testing and review process begins
```

## 📋 Development Guidelines

### Code Style
- **Black** for code formatting: `black src/`
- **isort** for import sorting: `isort src/`
- **flake8** for linting: `flake8 src/`
- **mypy** for type checking: `mypy src/`

### Testing
- The test suite lives in `tests/` and runs offline (deterministic hashed
  trigram embedder - no model downloads)
- Run locally with `pip install -e .[test]` then `pytest tests/ -q`
- Golden characterization tests pin the full metric output (atol=1e-12);
  regenerate `tests/golden/golden_cases.json` ONLY for intentional output
  changes, and review the JSON diff line by line
- CI (`.github/workflows/test.yml`) runs the suite on Python 3.10-3.13

### Documentation
- Update docstrings for new functions/classes
- Add examples for new features
- Update README if needed
- Website content includes copyright notices and proper attribution
- CLIP-CC dataset integration for benchmarking video models

## 🔄 Automated Workflows

### **Continuous Testing** (Every PR/Push)
When you submit a PR or push to main:
1. **Automated Testing**: Runs on Python 3.10-3.13
2. **Code Quality Checks**: Linting, formatting, type checking  
3. **Build Verification**: Ensures package builds correctly
4. **Fast Feedback**: Results in ~2-3 minutes

### **Release Publishing** (Manual Process)
Publishing is done manually by maintainers:
1. **Manual Trigger**: Go to Actions → "Build and Publish" → Run workflow
2. **Version Input**: Specify the desired version (e.g., 1.0.1)
3. **Environment Choice**: Select TestPyPI (testing) or PyPI (production)
4. **Package Building**: Creates wheel and source distributions
5. **Publishing**: Deploys to selected environment

## 💡 Commit Message Guidelines

### Format
```
<description>

[optional body]
[optional footer]
```

### Style
- Use clear, descriptive messages
- Start with a verb (add, fix, update, remove, etc.)
- Keep the first line under 50 characters when possible
- Use present tense ("add feature" not "added feature")

### Examples
```bash
# Good commit messages
add multilingual support for similarity metrics
refactor API to use async/await pattern
fix edge case in text alignment algorithm
fix bug in NAS calculation
add new scoring option
update documentation
improve code formatting
```

## 🏗️ Project Structure

```
vcs/
├── src/vcs/                 # Main package code (flat modules)
│   ├── __init__.py
│   └── scorer.py           # Main API
├── docs/                   # Documentation and website
│   ├── sphinx/             # Sphinx documentation source
│   ├── assets/             # Website assets (CSS, JS, videos)
│   ├── pages/              # Website pages
│   ├── widgets/            # Interactive widgets
│   └── index.html          # Main website
├── .github/workflows/      # CI/CD pipelines
│   ├── test.yml           # Continuous testing
│   ├── publish.yml        # Package publishing
│   └── deploy-docs.yml    # Documentation deployment
├── pyproject.toml         # Package configuration
├── DEPLOYMENT.md          # Deployment guide
└── CONTRIBUTING.md        # This file
```

## 🔧 Local Development Commands

```bash
# Install in development mode
pip install -e .[dev]

# Run code formatting
black src/
isort src/

# Run linting
flake8 src/

# Run type checking
mypy src/

# Build package locally
python -m build

# Test package installation
pip install dist/video-comprehension-score-*.whl
```

## 🐛 Debugging

### Version Issues
```bash
# Check current version in pyproject.toml
grep '^version = ' pyproject.toml

# List current git tags
git tag -l v*.*.* | sort -V

# Check package version
python -c "import vcs; print(vcs.__version__)"
```

### Build Issues
```bash
# Clean build artifacts
rm -rf dist/ build/ *.egg-info/

# Rebuild package
python -m build
```

## 📝 Release Process

### For Contributors
1. **Submit PR** with descriptive commit message
2. **Tests run automatically** (no publishing)
3. **Maintainer merges** when ready

### For Maintainers

#### **Manual Publishing** (Primary Method)
The standard two-step publishing process:

**Step 1: TestPyPI (Testing)**
1. Bump `version` in `pyproject.toml` and merge that commit
2. Go to **Actions** → **"Build and Publish"** → **"Run workflow"**
3. Enter the version (must match pyproject.toml) and select **"testpypi"**
4. **Test the package** on TestPyPI

**Step 2: PyPI (Production)**
1. After testing, go back to **Actions** → **"Build and Publish"**
2. Click **"Run workflow"**
3. Select **"pypi"** from dropdown
4. Click **"Run workflow"**
5. **Package is live** on PyPI

#### Version invariants
- The requested version must equal the committed `pyproject.toml` version;
  the workflow refuses to publish otherwise.
- The `vX.Y.Z` tag is pushed automatically after a successful PyPI publish.

## 🤝 Getting Help

- **Issues**: Create GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for urgent matters

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

**Happy Contributing!** 🎉