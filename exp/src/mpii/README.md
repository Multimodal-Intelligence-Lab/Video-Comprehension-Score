# MPII Dense Captions Evaluation Framework

A **production-ready, large-scale evaluation framework** for text generation models using BLEU, ROUGE, METEOR, and VCS (Video Comprehension Score) metrics on the **MPII Dense Captions Dataset**.

**🚀 Now optimized for large datasets (1,390+ files, 27,800+ computations) with enhanced checkpoint system for reliable processing.**

## 🎯 About VCS

**VCS (Video Comprehension Score)** is available as both:
- **GitHub Repository**: [https://github.com/hdubey-debug/vcs](https://github.com/hdubey-debug/vcs)
- **PyPI Package**: `pip install vcs`

This framework provides standardized evaluation pipelines for dense caption generation experiments.

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM (32GB+ recommended for large datasets)
- 50GB+ disk space (100GB+ for large-scale experiments)
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
   pip install transformers pandas numpy nltk rouge-score contractions wtpsplit
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
vcs/exp/
├── src/                               # Source code for evaluation frameworks
│   ├── mpii/                          # MPII evaluation scripts and configs
│   │   ├── scripts/                   # MPII evaluation scripts
│   │   │   ├── mpii_eval_addition.py          # Text addition experiments
│   │   │   ├── mpii_eval_authors.py           # Cross-author comparison
│   │   │   ├── mpii_eval_chronology.py        # Timeline sequencing
│   │   │   ├── mpii_eval_comparison.py        # General model evaluation
│   │   │   └── mpii_eval_deletion.py          # Text deletion experiments
│   │   ├── config/                    # Configuration files
│   │   │   ├── _base.yaml             # Base config (common settings)
│   │   │   ├── addition.yaml          # Addition experiment config
│   │   │   ├── authors.yaml           # Authors experiment config
│   │   │   ├── chronology.yaml        # Chronology experiment config
│   │   │   ├── comparison.yaml        # Comparison experiment config
│   │   │   └── deletion.yaml          # Deletion experiment config
│   │   ├── utils/                     # Shared utility modules
│   │   │   ├── core.py                # Core classes, constants, and utilities
│   │   │   └── traditional_metrics.py # BLEU, ROUGE, METEOR evaluation
│   │   ├── logs/                      # Execution logs (organized by experiment)
│   │   │   ├── addition/              # Addition experiment logs
│   │   │   ├── authors/               # Authors experiment logs
│   │   │   ├── chronology/            # Chronology experiment logs
│   │   │   ├── comparison/            # Comparison experiment logs
│   │   │   └── deletion/              # Deletion experiment logs
│   │   └── README.md                  # This file
│   └── vatex-eval/                    # VATEX evaluation framework
│       ├── scripts/                   # VATEX evaluation scripts
│       │   └── vatex_eval_ablation.py # VATEX ablation experiments
│       ├── config/                    # VATEX configuration files
│       │   └── ablation.yaml          # VATEX ablation config
│       ├── utils/                     # VATEX utility modules
│       │   ├── constants.py           # VATEX constants
│       │   └── core.py                # VATEX core utilities
│       └── logs/                      # VATEX execution logs
│           └── ablation/              # VATEX ablation logs
├── data/                              # Dataset storage
│   ├── mpii/                          # MPII Dense Captions data
│   │   ├── addition/
│   │   │   ├── aug_dataset/           # Addition experiments data
│   │   │   │   ├── beginning/         # Text added at beginning
│   │   │   │   ├── end/               # Text added at end
│   │   │   │   ├── middle/            # Text added in middle
│   │   │   │   └── random/            # Text added at random positions
│   │   │   └── raw_dataset/
│   │   ├── authors/
│   │   │   ├── aug_dataset/           # Cross-author comparison data
│   │   │   │   ├── author1.json
│   │   │   │   ├── author2.json
│   │   │   │   ├── author3.json
│   │   │   │   └── author4.json
│   │   │   └── raw_dataset/
│   │   ├── chronology/
│   │   │   ├── aug_dataset/           # Timeline sequencing data
│   │   │   │   ├── beginning/
│   │   │   │   └── end/
│   │   │   └── raw_dataset/
│   │   ├── comparison/
│   │   │   ├── aug_dataset/           # General evaluation data
│   │   │   │   ├── 1018.json
│   │   │   │   ├── 1020.json
│   │   │   │   └── 1026.json
│   │   │   └── raw_dataset/
│   │   └── deletion/
│   │       ├── aug_dataset/           # Deletion experiments data
│   │       │   ├── beginning/         # Text deleted from beginning
│   │       │   ├── end/               # Text deleted from end
│   │       │   ├── middle/            # Text deleted from middle
│   │       │   └── random/            # Text deleted from random positions
│   │       └── raw_dataset/
│   └── vatex-eval/                    # VATEX evaluation data
│       ├── candidates_list.pkl        # Generated captions
│       ├── gts_list.pkl               # Ground truth captions
│       ├── human_scores.pkl           # Human evaluation scores
│       └── vid_cans_score_dict.json   # Video-caption score mappings
├── results/                           # Evaluation results
│   ├── mpii/                          # MPII results organized by experiment
│   │   ├── addition/
│   │   ├── authors/
│   │   ├── chronology/
│   │   ├── comparison/
│   │   └── deletion/
│   └── vatex-eval/                    # VATEX evaluation results
│       └── ablation/                  # VATEX ablation results
```

## 🚀 Quick Start

### Navigate to the MPII Directory
```bash
cd vcs/exp/src/mpii
```

### Run Different Experiments

**Important**: All commands must be run from the `src/mpii/` directory.

#### Individual Experiments
```bash
# General comparison evaluation
python scripts/mpii_eval_comparison.py --config config/comparison.yaml

# Cross-author comparison
python scripts/mpii_eval_authors.py --config config/authors.yaml

# Text addition ablation
python scripts/mpii_eval_addition.py --config config/addition.yaml

# Text deletion ablation
python scripts/mpii_eval_deletion.py --config config/deletion.yaml

# Chronological sequencing
python scripts/mpii_eval_chronology.py --config config/chronology.yaml
```

### Enable Verbose Output
```bash
python scripts/mpii_eval_comparison.py --config config/comparison.yaml --verbose
```

### Override Experiment ID
```bash
# Override the experiment ID from config file
python scripts/mpii_eval_comparison.py --config config/comparison.yaml --experiment_id my_custom_id

# Can be combined with other options
python scripts/mpii_eval_comparison.py --config config/comparison.yaml --experiment_id resume_exp_123 --verbose
```

## 🔧 Configuration

The framework employs a hierarchical configuration system with base settings and experiment-specific overrides:

### **Configuration Architecture**
- **`_base.yaml`**: Common settings shared across all experiments
- **Individual configs**: Experiment-specific settings and paths
- **Automatic merging**: Runtime combination of base and experiment configurations

### Essential Settings (Must Update for Your System)

#### **Base Configuration (`config/_base.yaml`)**
```yaml
# Model paths - UPDATE THESE PATHS
models:
  nv_embed_path: "/path/to/nv-embed"
  sat_model: "sat-12l-sm"

# VCS computation settings
vcs:
  lct_values: [0, 1]                   # LCT variants to compute
  chunk_size: 1
  context_cutoff_value: 0.6
  context_window_control: 4.0

# Performance settings
processing:
  max_workers: 4                       # Adjust for your system
  resume_from_checkpoint: true

# Advanced settings
advanced:
  use_cuda: true                       # Enable CUDA if available
```

#### **Experiment Configuration Example**
```yaml
# Example: config/comparison.yaml
experiment:
  experiment_id: ""                # Auto-generate or specify for resume

paths:
  data_dir: "/path/to/data/mpii/comparison/aug_dataset"
  results_dir: "/path/to/results/mpii/comparison"
```

### **Configuration Benefits**
- ✅ **Efficient**: Minimal duplication across configuration files
- ✅ **Portable**: Copy 2 files for deployment to any machine
- ✅ **Maintainable**: Common settings centralized in base config
- ✅ **Compatible**: Standard command interface preserved

### **Deployment Setup**

For deployment to other machines:

```bash
# Copy required configuration files:
scp config/_base.yaml user@remote:/path/to/mpii/config/
scp config/comparison.yaml user@remote:/path/to/mpii/config/

# Run experiments normally:
python scripts/mpii_eval_comparison.py --config config/comparison.yaml
```

**Required files:**
- `_base.yaml`: Common settings (models, VCS parameters, processing options)
- `{experiment}.yaml`: Experiment-specific data paths and configurations

The framework automatically merges configurations at runtime without manual intervention.

### Available Experiments

1. **Comparison** (`config/comparison.yaml`)
   - General model evaluation on test cases
   - Standard BLEU, ROUGE, METEOR, VCS metrics
   - Output: `aggr_comp.csv`

2. **Authors** (`config/authors.yaml`)
   - Cross-author comparison experiments
   - Evaluates text generation across different authors
   - Output: Per-author CSV files and `aggr_comp_authors.csv`

3. **Addition** (`config/addition.yaml`)
   - Text addition ablation studies
   - Tests model performance when text is added at different positions
   - Supports both ablation and comparison metrics
   - Output: `aggr_ab_add_{position}.csv` and `aggr_comp_add_{position}.csv`

4. **Deletion** (`config/deletion.yaml`)
   - Text deletion ablation studies
   - Tests model robustness when text is removed
   - Supports both ablation and comparison metrics
   - Output: `aggr_ab_del_{position}.csv` and `aggr_comp_del_{position}.csv`

5. **Chronology** (`config/chronology.yaml`)
   - Timeline sequencing experiments
   - Evaluates chronological understanding
   - Output: `aggr_comp_chron_{position}.csv`

## 🛠️ Architecture & Utilities

The framework provides a production-ready architecture with shared utility modules optimized for large-scale datasets:

### `utils/core.py` - **Core Components**
- **Constants**: Shared constants for consistent processing
  - `PUNCTUATIONS`: Text processing punctuation handling
  - `DECIMAL_PRECISION`: Output formatting (3 decimal places)
  - `COMPARISON_METRICS_ORDER`: Standardized metric ordering
- **ConfigLoader**: Configuration loading and validation with automatic base config merging
- **ModelInitializer**: Centralized model initialization (SAT, NV-embed) with global scope management
- **CheckpointManager**: Production checkpoint system for large-scale processing
  - **Memory-efficient storage**: Metadata-only storage (~5MB vs 500MB+)
  - **Thread-safe operations**: Parallel processing with race condition prevention
  - **SHA-256 integrity validation**: Data corruption prevention and reliability
  - **Configuration consistency validation**: Resume compatibility checking
  - **Compressed storage (gzip)**: Reduced checkpoint file sizes
  - **Adaptive intervals**: Automatic frequency optimization based on processing speed
  - **Individual file storage**: Separate storage for memory efficiency
  - **Failed file tracking**: Intelligent handling of permanently failed files
  - **Processing statistics**: Performance metrics and success rate tracking
- **TextProcessor**: Shared text preprocessing functions
- **EmbeddingGenerator**: Centralized embedding generation

### `utils/traditional_metrics.py`
- BLEU, ROUGE, METEOR metric computation
- Standardized evaluation function interface
- Robust error handling and consistent metrics computation

## 📊 Data Formats

### General Format (Comparison, Addition, Deletion, Chronology)
```json
{
  "ground_truth": "Reference text describing the scene...",
  "iterations": {
    "1": "Generated text for iteration 1...",
    "2": "Generated text for iteration 2...",
    "3": "Generated text for iteration 3..."
  }
}
```

### Authors Format (Cross-author comparison)
```json
{
  "1": "Author's description for item 1...",
  "2": "Author's description for item 2...",
  "3": "Author's description for item 3..."
}
```

## 📈 Output Structure

Results are organized by experiment type with two levels of detail:

```
results/mpii/
├── comparison/
│   ├── individual_results/            # Per-file detailed results
│   │   ├── 1018.csv
│   │   ├── 1020.csv
│   │   └── 1026.csv
│   └── aggregated_results/            # Statistical summaries
│       └── aggr_comp.csv
├── authors/
│   ├── individual_results/            # Per-author results
│   │   ├── A1/                        # Author 1 results by item
│   │   │   ├── A1_1.csv
│   │   │   ├── A1_2.csv
│   │   │   └── A1_3.csv
│   │   └── ...
│   └── aggregated_results/            # Cross-author statistics
│       ├── A1.csv                     # Author 1 aggregated
│       ├── A2.csv                     # Author 2 aggregated
│       └── aggr_comp_authors.csv      # All authors combined
├── addition/
│   ├── individual_results/
│   │   ├── ablation/                  # Ablation study results
│   │   │   ├── beginning/
│   │   │   ├── end/
│   │   │   ├── middle/
│   │   │   └── random/
│   │   └── comparison/                # Traditional metric results
│   │       ├── beginning/
│   │       ├── end/
│   │       ├── middle/
│   │       └── random/
│   └── aggregated_results/
│       ├── ablation/                  # Ablation aggregated results
│       │   ├── aggr_ab_add_beginning.csv
│       │   ├── aggr_ab_add_end.csv
│       │   ├── aggr_ab_add_middle.csv
│       │   └── aggr_ab_add_random.csv
│       └── comparison/                # Traditional metrics aggregated
│           ├── aggr_comp_add_beginning.csv
│           ├── aggr_comp_add_end.csv
│           ├── aggr_comp_add_middle.csv
│           └── aggr_comp_add_random.csv
├── deletion/                          # Same structure as addition
└── chronology/
    ├── individual_results/
    │   ├── beginning/
    │   └── end/
    └── aggregated_results/
        ├── aggr_comp_chron_beginning.csv
        └── aggr_comp_chron_end.csv
```

### Result Format Examples

**Individual Results**: Detailed metrics for each iteration
```csv
Iteration,BLEU-1,BLEU-4,METEOR,ROUGE-1,ROUGE-4,ROUGE-L,ROUGE-Lsum,VCS_LCT0,VCS_LCT1
1,0.939,0.854,0.971,0.941,0.785,0.941,0.941,0.994,0.996
2,0.925,0.832,0.965,0.933,0.771,0.933,0.933,0.991,0.993
3,0.918,0.821,0.960,0.928,0.765,0.928,0.928,0.988,0.990
```

**Aggregated Results**: Statistical summaries in "mean ± std" format
```csv
Iteration,BLEU-1,BLEU-4,METEOR,ROUGE-1,ROUGE-4,ROUGE-L,ROUGE-Lsum,VCS_LCT0,VCS_LCT1
1,0.932 ± 0.007,0.843 ± 0.011,0.968 ± 0.003,0.937 ± 0.004,0.778 ± 0.007,0.937 ± 0.004,0.937 ± 0.004,0.993 ± 0.002,0.995 ± 0.001
2,0.924 ± 0.008,0.834 ± 0.012,0.964 ± 0.004,0.932 ± 0.005,0.772 ± 0.008,0.932 ± 0.005,0.932 ± 0.005,0.990 ± 0.003,0.993 ± 0.002
```

**Ablation Results**: VCS component analysis
```csv
Iteration,GAS,LAS,NAS-D,NAS-L,NAS,NAS+LAS(S),GAS+LAS(S),GAS+NAS-L(S),GAS+NAS-D(S),GAS+NAS(S),GAS+LAS(S)+NAS-D(S),GAS+LAS(S)+NAS-L(S),GAS+LAS(S)+(NAS-D+NAS-L)(S)
1,0.987,0.962,0.945,0.952,0.948,0.923,0.954,0.971,0.968,0.969,0.943,0.946,0.994
2,0.984,0.958,0.941,0.948,0.944,0.919,0.950,0.967,0.964,0.965,0.939,0.942,0.991
```

## 🚀 Large-Scale Processing Capabilities

The framework is optimized for production-scale datasets with the following capabilities:

### **Scalability Features**
- ✅ **Memory Efficiency**: Constant ~5MB usage regardless of dataset size (tested up to 27,800 computations)
- ✅ **Processing Capacity**: Handles 1,390+ files with 20+ test cases each
- ✅ **Interruption Recovery**: Safe resume from any point with data integrity validation
- ✅ **Parallel Processing**: Multi-threaded execution with configurable worker pools
- ✅ **Adaptive Performance**: Automatic checkpoint frequency optimization based on processing speed

### **Reliability Features**
- **Corruption Prevention**: SHA-256 checksums ensure data integrity
- **Configuration Validation**: Resume compatibility checking with incompatible settings prevention
- **Failed File Handling**: Intelligent tracking and skipping of permanently failed files
- **Performance Monitoring**: Real-time statistics and progress tracking
- **Graceful Degradation**: Robust error handling with automatic recovery

## 🏃‍♂️ Running Experiments

### Single Experiment
```bash
cd vcs/exp/src/mpii
python scripts/mpii_eval_comparison.py --config config/comparison.yaml
```

### Large-Scale Processing with Checkpointing
```bash
cd vcs/exp/src/mpii

# For large datasets, enable verbose mode to monitor progress
python scripts/mpii_eval_comparison.py --config config/comparison.yaml --verbose

# The checkpoint system automatically:
# - Saves progress every 2-3 minutes of processing time
# - Validates data integrity with checksums
# - Tracks failed files to avoid reprocessing
# - Optimizes checkpoint intervals based on processing speed
```

## 💾 Checkpointing System

The framework includes a robust checkpointing system for reliable recovery from interruptions during large-scale processing.

### **How Checkpointing Works**

The system automatically saves your progress and can resume from where it left off if interrupted.

#### **Default Behavior (Checkpointing Enabled)**
```yaml
# In all config files - checkpointing is enabled by default
processing:
  resume_from_checkpoint: true
  checkpoint_interval: 5
```

**What happens:**
- ✅ Progress is saved every few files automatically
- ✅ If you stop and restart, it continues from where you left off
- ✅ Data integrity is validated to prevent corruption
- ✅ Failed files are tracked and skipped on resume

### **Basic Usage**

#### **Starting a New Experiment**
```bash
# Simply run any experiment - checkpointing happens automatically
python scripts/mpii_eval_comparison.py --config config/comparison.yaml
```

#### **Resuming After Interruption**
If your experiment gets interrupted (timeout, cancellation, crash):

1. **Find your checkpoint:** Look for experiment ID in recent logs
   ```bash
   grep "Experiment ID" logs/comparison/mpii_comparison_*.log
   # Output: INFO: Experiment ID: eval_20250712_204623
   ```

2. **Update the specific experiment config with the experiment ID:**
   ```yaml
   # In the specific experiment config file (e.g., config/comparison.yaml)
   experiment:
     experiment_id: "eval_20250712_204623"  # Use the ID from logs
   ```

3. **Resume processing:**
   ```bash
   python scripts/mpii_eval_comparison.py --config config/comparison.yaml
   # Will automatically resume from checkpoint
   ```

### **Configuration Options**

#### **Enable Checkpointing (Default)**
```yaml
# In _base.yaml (global settings)
processing:
  resume_from_checkpoint: true    # Enable automatic resume
  checkpoint_interval: 5          # Save every 5 files

# In specific experiment config (e.g., config/comparison.yaml)
experiment:
  experiment_id: ""               # Auto-generate unique ID
```

#### **Disable Checkpointing**
```yaml
# In _base.yaml (global settings)
processing:
  resume_from_checkpoint: false   # Start fresh every time

# In specific experiment config (e.g., config/comparison.yaml)
experiment:
  experiment_id: ""               # Auto-generate unique ID
```

#### **Force Fresh Start (Even with Existing Checkpoints)**
```yaml
# In _base.yaml (global settings)
processing:
  resume_from_checkpoint: false   # Ignore any existing checkpoints

# In specific experiment config (e.g., config/comparison.yaml)
experiment:
  experiment_id: ""               # Generate new ID
```

### **Common Scenarios**

#### **Scenario 1: First Time Running**
- Just run the script - checkpointing works automatically
- No configuration changes needed

#### **Scenario 2: Job Got Interrupted**
1. Check logs for experiment ID
2. Put that ID in your specific experiment config file
3. Rerun the same command

#### **Scenario 3: Want to Start Over**
- Set `resume_from_checkpoint: false` in base config
- Run normally

#### **Scenario 4: Testing/Development**
- Set `resume_from_checkpoint: false` in base config for clean runs
- Use fixed experiment ID in specific experiment config for consistent testing

### **What Gets Saved**

The checkpoint system tracks:
- Which files have been completely processed
- Which files failed (to skip them on resume)
- Processing statistics and performance metrics
- Configuration validation to ensure consistency

### **Benefits**

#### **For Large Datasets**
- **Time Savings**: Hours of computation preserved across interruptions
- **Resource Efficiency**: No wasted computation from restarts
- **Reliability**: Automatic data integrity validation

#### **For Cluster/HPC Jobs**
- **Time Limits**: Resume when jobs hit time limits
- **Queue Flexibility**: Switch between different compute resources
- **Fault Tolerance**: Automatic recovery from node failures

### **Best Practices**

- **Production**: Always keep `resume_from_checkpoint: true`
- **Development**: Use `resume_from_checkpoint: false` for clean testing
- **Long Jobs**: Let the system auto-generate experiment IDs
- **Resume**: Specify experiment ID in the specific experiment config only when resuming interrupted jobs

### Batch Processing
```bash
cd vcs/exp/src/mpii

# Run all MPII experiments
for experiment in comparison authors addition deletion chronology; do
    echo "Running $experiment experiment..."
    python scripts/mpii_eval_${experiment}.py --config config/${experiment}.yaml --verbose
done

# Run with custom experiment ID prefix
for experiment in comparison authors addition deletion chronology; do
    echo "Running $experiment experiment..."
    python scripts/mpii_eval_${experiment}.py --config config/${experiment}.yaml --experiment_id batch_${experiment}_$(date +%Y%m%d) --verbose
done
```

### Monitor Progress (Enhanced Logging)
```bash
# Check logs for a specific experiment (organized by experiment type)
tail -f logs/comparison/mpii_comparison_*.log

# Check all logs
tail -f logs/*/mpii_*.log

# Monitor checkpoint status
ls -la results/mpii/comparison/.checkpoint_*

# Check processing statistics
grep "Checkpoint saved" logs/comparison/mpii_comparison_*.log
```

## 📊 Enhanced Logging System

The framework uses a structured logging system with experiment-specific organization:

### Log Structure
```
logs/
├── addition/                          # Addition experiment logs
│   └── mpii_addition_YYYYMMDD_HHMMSS.log
├── authors/                           # Authors experiment logs
│   └── mpii_authors_YYYYMMDD_HHMMSS.log
├── chronology/                        # Chronology experiment logs
│   └── mpii_chronology_YYYYMMDD_HHMMSS.log
├── comparison/                        # Comparison experiment logs
│   └── mpii_comparison_YYYYMMDD_HHMMSS.log
├── deletion/                          # Deletion experiment logs
│   └── mpii_deletion_YYYYMMDD_HHMMSS.log
```

### Log Contents
- Experiment configuration
- Processing progress with ETA
- Performance metrics
- Error messages with context
- Timing information
- Model initialization status
- Checkpointing information

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd vcs/exp/src/mpii
   # Verify VCS installation
   python -c "import vcs; print(vcs.__version__)"
   # Check shared utilities
   python -c "from utils.core import DECIMAL_PRECISION; print('Utils working')"
   ```

2. **Model Path Issues**
   ```bash
   # Update model paths in config files
   # Verify NV-Embed model location
   ls /path/to/nv-embed/
   # Check SAT model availability
   python -c "from wtpsplit import SaT; print('SAT available')"
   ```

3. **Memory Issues**
   ```bash
   # Reduce max_workers in config files
   # Check available RAM
   free -h
   # Monitor GPU memory
   nvidia-smi
   ```

4. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('wordnet')
   ```

5. **File Not Found Errors**
   ```bash
   # Verify data directory structure
   ls ../../data/mpii/comparison/aug_dataset/
   
   # Check config file paths
   grep -A5 "paths:" config/comparison.yaml
   
   # Check if _base.yaml exists
   ls config/_base.yaml
   
   # Verify utils module structure
   ls utils/
   ```

6. **Configuration Errors**
   ```bash
   # Verify configuration loading
   python -c "from utils.core import ConfigLoader; print('Config loading works')"
   
   # Check required config files exist
   ls config/_base.yaml config/comparison.yaml
   
   # Test configuration loading
   python -c "
   from utils.core import ConfigLoader
   config = ConfigLoader.load_config('config/comparison.yaml')
   print(f'Models section: {\"models\" in config}')
   print(f'Paths section: {\"paths\" in config}')
   "
   ```

7. **VCS Computation Errors**
   ```bash
   # Check for NV-embed model initialization errors
   grep "NV-embed model not initialized" logs/*/*.log
   
   # Verify VCS package installation
   python -c "import vcs; vcs.compute_vcs_score"
   ```

### Performance Optimization for Large Datasets

- **Parallel Processing**: Optimized worker allocation based on system resources
  - 1-2 workers: 8GB RAM systems
  - 4 workers: 16GB RAM systems  
  - 8+ workers: 32GB+ RAM systems
  - 16+ workers: 64GB+ RAM systems (for very large datasets)

- **Adaptive Checkpoint System**: Automatically enabled for all experiments
  - **Automatic optimization**: Checkpoint frequency adapts to processing speed
  - **Memory efficiency**: Constant ~5MB memory usage regardless of dataset size
  - **Integrity validation**: SHA-256 checksums prevent data corruption
  - **Configuration validation**: Prevents incompatible resume attempts
  - **Failed file tracking**: Intelligent handling of permanently failed files

- **Memory Management**: Optimized for large-scale processing
  - **Constant memory footprint**: No memory explosion with dataset growth
  - **Individual file results**: Separate storage prevents memory accumulation
  - **GPU memory monitoring**: Track model memory usage for large models

- **Storage Optimization**:
  - **Compressed checkpoints**: Gzip compression reduces storage requirements
  - **Distributed file storage**: Individual results stored separately for efficiency
  - **Automatic cleanup**: Checkpoints removed after successful completion

- **Production Features**:
  - **Thread-safe operations**: Safe parallel processing with locks
  - **Graceful error handling**: Robust recovery from transient failures
  - **Progress monitoring**: Real-time statistics and ETA estimation
  - **Performance tracking**: Detailed timing and resource usage logging

### Debugging VCS Issues

If you encounter VCS computation failures:

1. **Check Model Initialization**
   ```bash
   grep -i "model.*initialized" logs/*/mpii_*.log
   ```

2. **Verify VCS Configuration**
   ```bash
   grep -A10 "vcs:" config/*.yaml
   ```

3. **Test Individual Components**
   ```python
   # Test model initialization
   from utils.core import ModelInitializer
   ModelInitializer.initialize_sat()
   ModelInitializer.initialize_nvembed(config)
   
   # Test VCS computation
   import vcs
   result = vcs.compute_vcs_score(
       reference_text="test",
       generated_text="test",
       # ... other parameters
   )
   ```

## 🔬 Research Usage

### Reproducibility
All experiments are fully reproducible using the provided configuration files. Results include:
- Detailed logging with timestamps
- Checkpointing for interrupted runs
- Consistent random seeds
- Shared utility functions ensuring consistent preprocessing

### Metrics Computed
- **Traditional**: BLEU-1, BLEU-4, METEOR, ROUGE-1, ROUGE-4, ROUGE-L, ROUGE-Lsum
- **VCS**: Video Comprehension Score with configurable LCT values
- **Ablation**: GAS, LAS, NAS-D, NAS-L, and their combinations

### Framework Architecture - Production-Ready Design
- **🏗️ Modular Architecture**: 
  - **Hierarchical configuration**: Base config with experiment-specific overrides
  - **Shared utilities**: Common processing functions across all experiments
  - Consistent preprocessing and metrics computation
  - Industry-standard naming conventions (`traditional_metrics.py`, `core.py`)
  
- **🚀 Large-Scale Processing**:
  - Memory-efficient checkpoint system (5MB vs 1.5GB)
  - Thread-safe parallel processing with intelligent worker management
  - Adaptive performance optimization based on processing speed
  
- **🔒 Production-Grade Reliability**:
  - **Configuration validation**: Automatic base + experiment config merging
  - SHA-256 integrity validation prevents data corruption
  - Configuration consistency validation prevents resume errors  
  - Robust error handling with automatic recovery mechanisms
  - Failed file tracking and intelligent retry strategies
  
- **📊 Monitoring & Analytics**:
  - Real-time progress tracking with ETA estimation
  - Detailed performance metrics and resource usage logging
  - Processing statistics and success rate monitoring
  - Experiment-specific organized logging system
  
- **🎯 Deployment Features**:
  - **Minimal deployment**: Copy only `_base.yaml` + experiment config
  - **Zero configuration**: Automatic config merging at runtime
  - **Standard interface**: Consistent command structure across experiments

### Citation
If using this evaluation framework in your research, please cite the VCS paper:

```bibtex
@article{vcs2024,
  title={VCS: Video Comprehension Score for Evaluating Text Generation},
  author={Your Authors},
  journal={Your Journal},
  year={2024}
}
```

## 🤝 Contributing

1. Fork the VCS repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test with the evaluation framework
4. Test with individual experiments to ensure functionality
6. Submit pull request with detailed description

## 📄 License

This evaluation framework is part of the VCS project and follows the same license terms.

## 🆘 Support

For issues related to:
- **VCS framework**: Open an issue on the [VCS GitHub repository](https://github.com/hdubey-debug/vcs)
- **MPII experiments**: Check logs in `logs/` directory for detailed error information
- **Configuration**: Refer to the comments in configuration files for parameter explanations
- **Shared utilities**: Check `utils/` module documentation and source code

### Quick Support Checklist
1. Check experiment logs in `logs/{experiment}/`
2. Verify configuration files in `config/`
3. Test individual utilities in `utils/`
4. Monitor checkpoint status in `results/{experiment}/.checkpoint_*`
5. Check processing statistics with `grep "Checkpoint saved" logs/*/mpii_*.log`
6. Test with individual experiments using verbose output for diagnostics

---

## 🎯 Framework Summary

This production-ready MPII evaluation framework provides:

### ✅ **Large-Scale Capabilities**
- **Scalable**: Handles 1,390+ files with 27,800+ computations efficiently
- **Memory Efficient**: Constant 5MB usage regardless of dataset size
- **Reliable**: Robust checkpoint system with integrity validation and auto-recovery

### ✅ **Production Features**
- **Thread-Safe**: Parallel processing with intelligent worker management
- **Robust**: SHA-256 checksums, configuration validation, failed file tracking
- **Adaptive**: Automatic performance optimization based on processing speed
- **Monitored**: Real-time progress tracking with detailed analytics

### ✅ **Research-Ready**
- **Reproducible**: Consistent preprocessing and metrics across all experiments
- **Comprehensive**: BLEU, ROUGE, METEOR, VCS, and ablation metrics
- **Modular**: Clean architecture with shared utilities and industry-standard naming
- **Maintainable**: Hierarchical configuration system for easy management

### ✅ **Deployment-Ready**
- **Minimal deployment**: Copy `_base.yaml` + experiment config to any machine
- **Zero configuration**: Automatic config merging eliminates manual editing
- **Cross-platform**: Works identically across different compute environments

**🚀 Ready for production-scale dense caption evaluation with streamlined deployment and reliable checkpoint recovery.**