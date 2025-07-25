# MPII Ablation Study Framework

A **production-ready ablation study framework** for analyzing VCS (Video Comprehension Score) component contributions on the **MPII Dense Captions Dataset**.

**🔬 Comprehensive VCS component analysis with two complementary ablation approaches and robust checkpoint-based processing.**

## 🎯 About VCS

**VCS (Video Comprehension Score)** is available as both:
- **GitHub Repository**: [https://github.com/hdubey-debug/vcs](https://github.com/hdubey-debug/vcs)
- **PyPI Package**: `pip install vcs`

This framework provides standardized ablation analysis pipelines for understanding VCS component contributions to overall performance.

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM (32GB+ recommended for large datasets)
- 20GB+ disk space
- Multi-core CPU (4+ cores recommended for parallel processing)

### Environment Setup

1. **Install VCS package**
   ```bash
   # Option A: Install from PyPI
   pip install vcs

   # Option B: Install from GitHub (development mode)
   git clone https://github.com/hdubey-debug/vcs.git
   cd vcs
   pip install -e ".[dev]"
   pre-commit install
   ```

2. **Install additional dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers pandas numpy nltk contractions wtpsplit
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('wordnet')
   ```

## 📁 Directory Structure

```
vcs/ablation/
├── src/                               # Source code for ablation framework
│   ├── scripts/                       # Ablation evaluation scripts
│   │   └── mpii_eval_ablation.py     # Main ablation study script
│   ├── config/                        # Configuration files
│   │   └── mpii_eval_ablation.yaml   # Ablation experiment config
│   ├── utils/                         # Shared utility modules
│   │   └── core.py                   # Core classes, constants, and utilities
│   └── logs/                          # Execution logs
│       └── mpii_ablation_*.log       # Timestamped log files
├── data/                              # Test case data (JSON files)
│   ├── 1.json                        # Individual test case files
│   ├── 126.json                      # Example test cases
│   ├── 704.json
│   └── 947.json
├── results/                           # Ablation study results
│   ├── ablation_1/                   # All VCS metrics results
│   │   ├── individual_results/       # Per-file detailed results
│   │   │   ├── 126.csv
│   │   │   ├── 704.csv
│   │   │   └── 947.csv
│   │   └── aggregated_results/       # Statistical summaries
│   │       └── aggregated_results.csv
│   └── ablation_2/                   # Custom ablation metrics results
│       ├── individual_results/       # Per-file detailed results
│       └── aggregated_results/       # Statistical summaries
└── README.md                          # This file
```

## 🚀 Quick Start

### Navigate to the Ablation Directory
```bash
cd vcs/ablation/src
```

### Basic Usage
```bash
# Run complete ablation study (both ablation types)
python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml
```

### Advanced Usage
```bash
# Enable verbose output for monitoring
python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml --verbose

# Override experiment ID for resuming interrupted runs
python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml --experiment_id ablation_20250723_163441

# Combine options
python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml --experiment_id my_ablation_study --verbose
```

## 🔧 Configuration

Update `config/mpii_eval_ablation.yaml` with your system paths:

```yaml
# Model paths - UPDATE THESE PATHS
models:
  nv_embed_path: "/path/to/nv-embed"
  sat_model: "sat-12l-sm"

# Data and output paths
paths:
  data_dir: "../data"                  # JSON test case files
  results_dir: "../results"           # Output directory
  logs_dir: "logs"                     # Log files directory

# VCS computation settings
vcs:
  chunk_size: 1
  context_cutoff_value: 0.6
  context_window_control: 4.0
  lct: 0
  return_all_metrics: true
  return_internals: false

# Performance settings
processing:
  max_workers: 12                      # Adjust for your system
  resume_from_checkpoint: true
  checkpoint_interval: 5

# Ablation study configuration
ablation:
  run_ablation_1: true                 # All VCS metrics
  run_ablation_2: true                 # Custom ablation metrics
```

## 📊 Data Format

Input JSON format for test cases:
```json
{
  "ground_truth": "Reference description of the scene...",
  "categories": [
    {
      "category_name": "Test Category",
      "test_cases": [
        {
          "id": "1",
          "name": "Test Case 1",
          "description": "Generated text for evaluation..."
        },
        {
          "id": "2", 
          "name": "Test Case 2",
          "description": "Another generated text..."
        }
      ]
    }
  ]
}
```

## 🔬 Ablation Study Types

### **Ablation 1: All VCS Metrics**
- **Purpose**: Comprehensive analysis of all VCS components
- **Method**: Uses `return_all_metrics=True` to extract all VCS metrics
- **Output Metrics**:
  - **Precision/Recall NAS-D**: Narrative Arc Score (Dependency) components
  - **Precision/Recall NAS-L**: Narrative Arc Score (Lexical) components  
  - **NAS-D, NAS-L, NAS**: Narrative Arc Scores
  - **Precision/Recall LAS**: Local Alignment Score components
  - **LAS**: Local Alignment Score
  - **GAS**: Global Alignment Score
  - **GAS-LAS-Scaled**: Scaled combination
  - **VCS**: Complete Video Comprehension Score

### **Ablation 2: Custom Ablation Metrics**
- **Purpose**: Analysis of scaled metric combinations
- **Method**: Custom computation of VCS component interactions
- **Output Metrics**:
  - **Base Components**: GAS, LAS, NAS-D, NAS-L, NAS
  - **Binary Combinations**: NAS+LAS(S), GAS+LAS(S), GAS+NAS-L(S), GAS+NAS-D(S), GAS+NAS(S)
  - **Triple Combinations**: GAS+LAS(S)+NAS-D(S), GAS+LAS(S)+NAS-L(S)
  - **Complete VCS**: GAS+LAS(S)+(NAS-D+NAS-L)(S)

## 📈 Output Structure

Results are organized by ablation type with detailed analysis:

```
results/
├── ablation_1/                       # All VCS metrics analysis
│   ├── individual_results/           # Per-file detailed results
│   │   ├── 126.csv                   # Metrics as rows, test cases as columns
│   │   ├── 704.csv
│   │   └── 947.csv
│   └── aggregated_results/           # Cross-file statistical analysis
│       └── aggregated_results.csv    # Mean ± std across all files
└── ablation_2/                       # Custom ablation metrics analysis
    ├── individual_results/           # Per-file detailed results
    └── aggregated_results/           # Cross-file statistical analysis
```

### Individual Results Format
```csv
metric,Test Case 1,Test Case 2,Test Case 3
GAS,0.987,0.984,0.981
LAS,0.962,0.958,0.955
NAS-D,0.945,0.941,0.938
VCS,0.994,0.991,0.988
```

### Aggregated Results Format
```csv
Metric,Test Case 1,Test Case 2,Test Case 3
GAS,0.984 ± 0.003,0.981 ± 0.004,0.978 ± 0.005
LAS,0.958 ± 0.007,0.955 ± 0.008,0.952 ± 0.009
NAS-D,0.941 ± 0.006,0.938 ± 0.007,0.935 ± 0.008
VCS,0.991 ± 0.002,0.988 ± 0.003,0.985 ± 0.004
```

## 🛠️ Architecture & Utilities

The framework provides a production-ready architecture with shared utility modules:

### `src/utils/core.py` - **Core Components**
- **Constants**: Shared constants for consistent processing
  - `PUNCTUATIONS`: Text processing punctuation handling  
  - `DECIMAL_PRECISION`: Output formatting (3 decimal places)
  - `ABLATION_1_METRICS_ORDER`: Standardized ablation 1 metric ordering
  - `ABLATION_2_METRICS_ORDER`: Standardized ablation 2 metric ordering
  - `DEFAULT_SEGMENTER_FUNCTION`: Default text segmentation approach ("sat")
- **ConfigLoader**: Configuration loading and validation
- **ModelInitializer**: Centralized model initialization (NV-embed, SAT)
- **CheckpointManager**: Production checkpoint system for large-scale processing
  - **Memory-efficient storage**: Metadata-only storage (~5MB vs 500MB+)
  - **Thread-safe operations**: Parallel processing with race condition prevention
  - **SHA-256 integrity validation**: Data corruption prevention and reliability
  - **Configuration consistency validation**: Resume compatibility checking
  - **Compressed storage (gzip)**: Reduced checkpoint file sizes
  - **Adaptive intervals**: Automatic frequency optimization based on processing speed
- **TextProcessor**: Text preprocessing and segmentation (SAT-based)
- **EmbeddingGenerator**: Centralized embedding generation (NV-embed)
- **AblationUtils**: Ablation-specific utilities including custom metric computation

## 💾 Checkpointing System

The framework includes a robust checkpointing system for reliable recovery from interruptions.

### **How Checkpointing Works**

#### **Default Behavior (Checkpointing Enabled)**
```yaml
# In config file - checkpointing is enabled by default
processing:
  resume_from_checkpoint: true
  checkpoint_interval: 5
```

**What happens:**
- ✅ Progress is saved every few test cases automatically
- ✅ If interrupted, restarts continue from where they left off
- ✅ Data integrity is validated to prevent corruption
- ✅ Failed test cases are tracked and skipped on resume

### **Basic Usage**

#### **Starting a New Ablation Study**
```bash
# Simply run the script - checkpointing happens automatically
python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml
```

#### **Resuming After Interruption**
If your ablation study gets interrupted:

1. **Find your experiment ID:** Look in recent logs
   ```bash
   grep "Experiment ID" logs/mpii_ablation_*.log
   # Output: Starting fresh evaluation with new experiment ID: ablation_20250723_163441
   ```

2. **Resume with the experiment ID:**
   ```bash
   python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml --experiment_id ablation_20250723_163441
   # Will automatically resume from checkpoint
   ```

### **Configuration Options**

#### **Enable Checkpointing (Default)**
```yaml
processing:
  resume_from_checkpoint: true    # Enable automatic resume
  checkpoint_interval: 5          # Save every 5 test cases
```

#### **Disable Checkpointing**
```yaml
processing:
  resume_from_checkpoint: false   # Start fresh every time
```

## 🏃‍♂️ Running Ablation Studies

### Single Ablation Study
```bash
cd vcs/ablation/src
python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml
```

### With Progress Monitoring
```bash
# For monitoring progress, enable verbose mode
python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml --verbose

# Monitor logs in real-time
tail -f logs/mpii_ablation_*.log
```

### Monitor Progress
```bash
# Check logs for specific experiment
tail -f logs/mpii_ablation_*.log

# Check checkpoint status
ls -la results/.checkpoint_*

# Check processing statistics
grep "Checkpoint saved" logs/mpii_ablation_*.log
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd vcs/ablation/src
   # Verify VCS installation
   python -c "import vcs; print('VCS available')"
   # Check shared utilities
   python -c "from utils.core import DECIMAL_PRECISION; print('Utils working')"
   ```

2. **Model Path Issues**
   ```bash
   # Update model paths in config file
   # Verify NV-Embed model location
   ls /path/to/nv-embed/
   # Check SAT model availability
   python -c "from wtpsplit import SaT; print('SAT available')"
   ```

3. **Memory Issues**
   ```bash
   # Reduce max_workers in config file
   # Check available RAM
   free -h
   # Monitor GPU memory
   nvidia-smi
   ```

4. **Configuration Errors**
   ```bash
   # Verify configuration loading
   python -c "from utils.core import ConfigLoader; print('Config loading works')"
   
   # Test configuration loading
   python -c "
   from utils.core import ConfigLoader
   config = ConfigLoader.load_config('config/mpii_eval_ablation.yaml')
   print(f'Models section: {\"models\" in config}')
   print(f'Paths section: {\"paths\" in config}')
   "
   ```

5. **VCS Computation Errors**
   ```bash
   # Check for model initialization errors
   grep "model.*initialized" logs/mpii_ablation_*.log
   
   # Verify VCS package installation
   python -c "import vcs; print('VCS computation available')"
   ```

### Performance Optimization

- **Parallel Processing**: Optimized worker allocation based on system resources
  - 1-2 workers: 8GB RAM systems
  - 4-8 workers: 16GB RAM systems  
  - 12+ workers: 32GB+ RAM systems

- **Memory Management**: Optimized for ablation studies
  - **Constant memory footprint**: No memory explosion with dataset growth
  - **Individual file results**: Separate storage prevents memory accumulation
  - **GPU memory monitoring**: Track model memory usage

## 🔬 Research Usage

### Reproducibility
All ablation experiments are fully reproducible using the provided configuration files. Results include:
- Detailed logging with timestamps
- Checkpointing for interrupted runs
- Consistent random seeds
- Shared utility functions ensuring consistent preprocessing

### Metrics Analysis
- **Ablation 1**: Comprehensive VCS component analysis with precision/recall breakdown
- **Ablation 2**: Scaled metric combination analysis for understanding component interactions
- **Statistical Analysis**: Mean ± std across multiple test cases for robust evaluation

### Framework Architecture - Production-Ready Design
- **🏗️ Modular Architecture**: 
  - **Ablation-specific utilities**: Custom metric computation and analysis functions
  - **Shared utilities**: Common processing functions with MPII-style consistency
  - **Industry-standard naming**: Clear separation of ablation types and utilities
  
- **🚀 Large-Scale Processing**:
  - Memory-efficient checkpoint system with integrity validation
  - Thread-safe parallel processing optimized for ablation studies
  - Adaptive performance optimization based on processing speed
  
- **🔒 Production-Grade Reliability**:
  - **Configuration validation**: Automatic config loading and validation
  - SHA-256 integrity validation prevents data corruption
  - Configuration consistency validation prevents resume errors  
  - Robust error handling with automatic recovery mechanisms
  
- **📊 Monitoring & Analytics**:
  - Real-time progress tracking with ETA estimation
  - Detailed performance metrics and resource usage logging
  - Ablation-specific organized logging system

## 🎯 Framework Summary

This production-ready MPII ablation framework provides:

### ✅ **Ablation Study Capabilities**
- **Comprehensive Analysis**: Two complementary ablation approaches for complete VCS understanding
- **Statistical Rigor**: Mean ± std analysis across multiple test cases for robust conclusions
- **Component Isolation**: Individual and combined metric analysis for detailed insights

### ✅ **Production Features**
- **Thread-Safe**: Parallel processing with intelligent worker management
- **Robust**: SHA-256 checksums, configuration validation, failed case tracking
- **Adaptive**: Automatic performance optimization based on processing speed
- **Monitored**: Real-time progress tracking with detailed analytics

### ✅ **Research-Ready**
- **Reproducible**: Consistent preprocessing and metrics across all ablation studies
- **Comprehensive**: Complete VCS component analysis with precision/recall breakdown
- **Modular**: Clean architecture with ablation-specific utilities
- **Maintainable**: Clear configuration system for easy experiment management

### ✅ **Deployment-Ready**
- **Minimal setup**: Single config file for complete ablation study configuration
- **Zero manual intervention**: Automatic checkpoint recovery and result organization
- **Cross-platform**: Works identically across different compute environments

**🔬 Ready for production-scale VCS ablation analysis with comprehensive component breakdown and reliable checkpoint recovery.**

## 🤝 Contributing

1. Fork the VCS repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test with the ablation framework
4. Test with individual ablation studies to ensure functionality
5. Submit pull request with detailed description

## 📄 License

This ablation framework is part of the VCS project and follows the same license terms.

## 🆘 Support

For issues related to:
- **VCS framework**: Open an issue on the [VCS GitHub repository](https://github.com/hdubey-debug/vcs)
- **Ablation experiments**: Check logs in `logs/` directory for detailed error information
- **Configuration**: Refer to the comments in `config/mpii_eval_ablation.yaml` for parameter explanations
- **Shared utilities**: Check `utils/core.py` module documentation and source code

### Quick Support Checklist
1. Check experiment logs in `logs/mpii_ablation_*.log`
2. Verify configuration file in `config/mpii_eval_ablation.yaml`
3. Test individual utilities in `utils/core.py`
4. Monitor checkpoint status in `results/.checkpoint_*`
5. Check processing statistics with `grep "Checkpoint saved" logs/mpii_ablation_*.log`
6. Test with verbose output for diagnostics

---

**🔬 Ready for comprehensive VCS ablation analysis with production-grade reliability and detailed component insights.**