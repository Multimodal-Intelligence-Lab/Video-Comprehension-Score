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

# Install development dependencies
# Note: PyTorch is required but not auto-installed to avoid conflicts
pip install torch  # Install PyTorch first (see https://pytorch.org)
pip install -e .[dev]
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
- Testing is integrated into CI/CD workflows
- All functionality is tested automatically via comprehensive CI tests
- Tests include package validation, API testing, and integration testing
- No separate test files - testing occurs in `.github/workflows/test.yml`

### Documentation
- Update docstrings for new functions/classes
- Add examples for new features
- Update README if needed
- Website content includes copyright notices and proper attribution
- CLIP-CC dataset integration for benchmarking video models

## 🔄 Automated Workflows

### **Continuous Testing** (Every PR/Push)
When you submit a PR or push to main:
1. **Automated Testing**: Runs on Python 3.11+
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
add new visualization feature
update documentation
improve code formatting
```

## 🏗️ Project Structure

```
vcs/
├── src/vcs/                 # Main package code
│   ├── __init__.py
│   ├── _metrics/           # Core metrics implementations
│   ├── _visualize_vcs/     # Visualization components
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
1. Go to **Actions** → **"Build and Publish"**
2. Click **"Run workflow"**
3. Select **"testpypi"** from dropdown
4. Click **"Run workflow"**
5. **Test the package** on TestPyPI

**Step 2: PyPI (Production)**
1. After testing, go back to **Actions** → **"Build and Publish"**
2. Click **"Run workflow"**
3. Select **"pypi"** from dropdown
4. Click **"Run workflow"**
5. **Package is live** on PyPI

#### **Alternative: GitHub Release Method**
You can also create GitHub releases for automatic TestPyPI publishing:
1. **Create GitHub Release**:
   - Go to GitHub → Releases → "Create a new release"
   - Tag: `v1.2.0`
   - Title: `Release v1.2.0`
   - Description: What changed
   - Click "Publish release"
2. **Result**: Automatically publishes to TestPyPI only

#### **Alternative: Direct Tag Push**
```bash
# Create and push version tag
git tag -a v1.2.0 -m "Release v1.2.0: Add new features"
git push origin v1.2.0
# → Automatic TestPyPI publishing only
```

## 🤝 Getting Help

- **Issues**: Create GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for urgent matters

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

**Happy Contributing!** 🎉