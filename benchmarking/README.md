# VLM Evaluation Framework on CLIP-CC Dataset

A **production-ready, scalable evaluation framework** for Vision-Language Models (VLMs) using VCS (Video Comprehension Score) metrics on the **CLIP-CC dataset**. This framework provides comprehensive evaluation of VLM-generated video descriptions against human-written ground truth summaries.

**🚀 Optimized for large-scale VLM evaluation with parallel processing, checkpointing, and comprehensive VCS analysis across multiple configurations.**

## 🎯 About VCS and CLIP-CC

**VCS (Video Comprehension Score)** is available as both:
- **GitHub Repository**: [https://github.com/hdubey-debug/vcs](https://github.com/hdubey-debug/vcs)
- **PyPI Package**: `pip install vcs`

**CLIP-CC Dataset** provides:
- 200 movie clips (1 minute 30 seconds each)
- Human-written detailed summaries
- YouTube video links for each clip
- Professional-quality ground truth annotations

This framework evaluates VLM predictions against CLIP-CC ground truth using multiple VCS configurations for comprehensive analysis.

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM (32GB+ recommended for large VLM evaluations)
- 20GB+ disk space (50GB+ for multiple model evaluations)
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
   pip install transformers pandas numpy nltk rouge-score contractions wtpsplit datasets
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
vcs/vlm_eval/
├── src/                                    # Source code for VLM evaluation
│   ├── config/                             # Configuration files
│   │   └── clipcc_eval_vlms.yaml          # Main evaluation configuration
│   ├── scripts/                           # Evaluation scripts
│   │   └── clipcc_eval_vlms.py            # Main VLM evaluation script
│   ├── utils/                             # Shared utility modules
│   │   └── core.py                        # Core classes and utilities
│   └── logs/                              # Execution logs
├── data/                                  # Dataset and model predictions
│   ├── clip-cc/                           # CLIP-CC dataset storage
│   │   └── clip_cc_dataset.json           # Downloaded CLIP-CC dataset
│   └── models/                            # VLM prediction files
│       ├── internvl.json                  # InternVL model predictions
│       ├── llava_next.json                # LLaVA-Next predictions
│       ├── videollama3.json               # VideoLLaMA3 predictions
│       ├── oryx.json                      # Oryx model predictions
│       ├── mplug.json                     # mPLUG-DocOwl predictions
│       ├── timechat.json                  # TimeChat predictions
│       ├── long_va.json                   # Long-VA predictions
│       ├── llava_one_vision.json          # LLaVA-OneVision predictions
│       ├── minicpm.json                   # MiniCPM-V predictions
│       ├── ts_llava.json                  # TS-LLaVA predictions
│       ├── videochatflash.json            # VideoChatFlash predictions
│       └── vilamp.json                    # VILAMP predictions
├── results/                               # Evaluation results
│   ├── individual_results/                # Per-model detailed results
│   │   ├── internvl.csv                   # InternVL individual results
│   │   ├── llava_next.csv                 # LLaVA-Next individual results
│   │   └── ...                            # Other model results
│   ├── aggregated_results/                # Statistical summaries
│   │   └── aggregated_results.csv         # All models aggregated
│   └── logs/                              # Processing logs
└── README.md                              # This file
```

## 📦 Dataset Setup

### Download CLIP-CC Dataset from Hugging Face

The CLIP-CC dataset is hosted on Hugging Face and can be downloaded using the `datasets` library:

```bash
# Navigate to VLM evaluation directory
cd vcs/vlm_eval

# Ensure datasets library is installed
pip install datasets

# Download CLIP-CC dataset
python -c "
from datasets import load_dataset
import json
from pathlib import Path

# Load the dataset from Hugging Face
print('Loading CLIP-CC dataset from Hugging Face...')
dataset = load_dataset('IVSL-SDSU/Clip-CC')

# Display dataset information
print(f'Dataset structure: {dataset}')
print(f'Total samples: {len(dataset[\"train\"])}')
print(f'Features: {dataset[\"train\"].column_names}')

# Convert to JSON format and save
output_dir = Path('data/clip-cc')
output_dir.mkdir(parents=True, exist_ok=True)

data_list = []
for item in dataset['train']:
    data_list.append(dict(item))

output_file = output_dir / 'clip_cc_dataset.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, indent=2, ensure_ascii=False)

print(f'Dataset saved to {output_file}')
print(f'Total samples downloaded: {len(data_list)}')
"
```

**Alternative: Manual Download**
```bash
# Using Hugging Face CLI (if preferred)
pip install huggingface_hub
huggingface-cli download IVSL-SDSU/Clip-CC --repo-type dataset --local-dir data/clip-cc/
```

### Verify Dataset Download

```bash
# Check dataset structure
ls -la data/clip-cc/

# Verify dataset content
python -c "
import json
with open('data/clip-cc/clip_cc_dataset.json', 'r') as f:
    data = json.load(f)
print(f'Successfully loaded {len(data)} samples')
print(f'Sample keys: {list(data[0].keys())}')
print(f'First sample ID: {data[0][\"id\"]}')
"
```

### Model Predictions Format

VLM predictions should be stored as JSON files in `data/models/` directory. Two formats are supported:

**Format 1: List of dictionaries**
```json
[
  {
    "id": "001",
    "summary": "Model-generated description of the video...",
    "file_link": "https://www.youtube.com/watch?v=..."
  }
]
```

**Format 2: Dictionary mapping (most common)**
```json
{
  "001": "Model-generated description for video 001...",
  "002": "Model-generated description for video 002...",
  "003": "Model-generated description for video 003..."
}
```

## 🚀 Quick Start

### Navigate to VLM Evaluation Directory
```bash
cd vcs/vlm_eval/src/scripts
```

### Run VLM Evaluation

**Basic evaluation with default settings:**
```bash
python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml
```

**With custom experiment ID:**
```bash
python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml --experiment_id my_vlm_experiment
```

**With verbose output for detailed monitoring:**
```bash
python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml --experiment_id detailed_analysis
```

### Advanced Usage Options
```bash
# Override experiment ID from config file
python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml --experiment_id custom_evaluation_001

# Combine options for detailed tracking
python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml --experiment_id vlm_comparison_study
```

## 🔧 Configuration

The framework uses a single comprehensive configuration file for all VLM evaluation settings:

### **Configuration File (`config/clipcc_eval_vlms.yaml`)**

```yaml
# ============================================================================
# VLM EVALUATION ON CLIP-CC DATASET CONFIGURATION
# ============================================================================

# Experiment settings
experiment:
  experiment_id: ""                        # Auto-generate or specify for resume

# Model configuration
models:
  nv_embed_path: "/path/to/nv-embed"      # UPDATE: Path to NV-Embed model
  sat_model: "sat-12l-sm"                 # SAT model for text segmentation

# VCS computation settings
vcs:
  chunk_sizes: [1, 2]                     # Evaluate both chunk sizes
  lct_values: [0, 1]                      # LCT variants to compute
  context_cutoff_value: 0.6
  context_window_control: 4.0
  return_internals: false

# Parallel processing
processing:
  max_workers: 12                         # Adjust for your system
  checkpoint_interval: 10                 # Save every N models
  resume_from_checkpoint: true

# Model parameters
embedding:
  batch_size: 8
  max_length: 32768
  instruction: ""

# Output formatting
output:
  decimal_precision: 3                    # Number of decimal places

# Logging
logging:
  level: "INFO"
  verbose: false

# Advanced settings
advanced:
  use_cuda: true                          # Enable CUDA if available

# Paths (UPDATE THESE PATHS)
paths:
  clipcc_data_dir: "/path/to/vcs/vlm_eval/data/clip-cc"
  models_data_dir: "/path/to/vcs/vlm_eval/data/models"
  results_dir: "/path/to/vcs/vlm_eval/results"
  individual_results_dir: "individual_results"
  aggregated_results_dir: "aggregated_results"
  logs_dir: "logs"
```

### Essential Settings (Must Update for Your System)

1. **Update Model Paths**
   ```yaml
   models:
     nv_embed_path: "/your/path/to/nv-embed"  # Download from HuggingFace
   ```

2. **Update Data Paths**
   ```yaml
   paths:
     clipcc_data_dir: "/absolute/path/to/vcs/vlm_eval/data/clip-cc"
     models_data_dir: "/absolute/path/to/vcs/vlm_eval/data/models"
     results_dir: "/absolute/path/to/vcs/vlm_eval/results"
   ```

3. **Adjust Performance Settings**
   ```yaml
   processing:
     max_workers: 4                         # Based on your system (2-16)
   ```

## 🎯 VCS Evaluation Configurations

The framework evaluates VLMs using multiple VCS configurations for comprehensive analysis:

### **Evaluation Matrix**
- **Chunk Sizes**: 1, 2
- **LCT Values**: 0, 1
- **Total Configurations**: 4 per model

### **Output Metrics**
For each VLM, the framework computes:
1. **VCS (chunk_size=1, LCT=0)**: Basic semantic alignment
2. **VCS (chunk_size=1, LCT=1)**: Enhanced context understanding
3. **VCS (chunk_size=2, LCT=0)**: Multi-segment analysis
4. **VCS (chunk_size=2, LCT=1)**: Advanced multi-segment with context

## 📊 Output Structure

Results are organized with two levels of detail for comprehensive analysis:

### **Individual Results** (`individual_results/`)
Detailed VCS scores for each video clip, per model:

```csv
id,VCS (chunk_size=1, LCT=0),VCS (chunk_size=1, LCT=1),VCS (chunk_size=2, LCT=0),VCS (chunk_size=2, LCT=1),file_link
001,0.892,0.905,0.876,0.891,https://www.youtube.com/watch?v=...
002,0.834,0.847,0.823,0.839,https://www.youtube.com/watch?v=...
003,0.901,0.914,0.885,0.898,https://www.youtube.com/watch?v=...
```

### **Aggregated Results** (`aggregated_results/`)
Statistical summaries across all models in "mean ± std" format:

```csv
model_name,VCS (chunk_size=1, LCT=0),VCS (chunk_size=1, LCT=1),VCS (chunk_size=2, LCT=0),VCS (chunk_size=2, LCT=1)
internvl,0.863 ± 0.045,0.876 ± 0.043,0.851 ± 0.048,0.864 ± 0.046
llava_next,0.841 ± 0.052,0.854 ± 0.050,0.829 ± 0.055,0.842 ± 0.053
videollama3,0.798 ± 0.063,0.811 ± 0.061,0.786 ± 0.066,0.799 ± 0.064
```

### **Log Files** (`results/logs/`)
Detailed execution logs with:
- Processing progress and timing
- Model initialization status  
- Error messages and warnings
- Performance statistics
- VCS computation details

## 🚀 Large-Scale Processing Capabilities

The framework is optimized for production-scale VLM evaluation with the following capabilities:

### **Scalability Features**
- ✅ **Multi-Model Support**: Evaluate 10+ VLMs simultaneously
- ✅ **Parallel Processing**: Configurable worker pools for optimal performance
- ✅ **Memory Efficiency**: Constant memory usage regardless of model count
- ✅ **Interruption Recovery**: Safe resume from any point with data integrity validation
- ✅ **Adaptive Performance**: Automatic optimization based on processing speed

### **Reliability Features**
- **Data Integrity**: Validation ensures consistent evaluation across models
- **Error Handling**: Robust processing with automatic error recovery
- **Progress Tracking**: Real-time monitoring with detailed statistics
- **Failed Model Handling**: Intelligent tracking and skipping of problematic models
- **Configuration Validation**: Ensures evaluation consistency

## 🏃‍♂️ Running Evaluations

### Single Model Evaluation
```bash
cd vcs/vlm_eval/src/scripts

# Evaluate all models in data/models/
python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml
```

### Large-Scale Evaluation with Progress Monitoring
```bash
cd vcs/vlm_eval/src/scripts

# Run with detailed progress tracking
python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml --experiment_id large_scale_eval
```

### Monitor Progress
```bash
# Check processing logs
tail -f ../../../results/logs/clipcc_eval_*.log

# Monitor individual results as they're generated
ls -la ../../../results/individual_results/

# Check aggregated results
cat ../../../results/aggregated_results/aggregated_results.csv
```

## 💾 Checkpointing System

The framework includes an intelligent checkpointing system for reliable recovery during large-scale evaluations.

### **How Checkpointing Works**

The system automatically saves progress and can resume from interruptions.

#### **Default Behavior (Checkpointing Enabled)**
```yaml
# In config file - checkpointing is enabled by default
processing:
  resume_from_checkpoint: true
  checkpoint_interval: 10
```

**What happens:**
- ✅ Progress saved after every few models automatically
- ✅ If interrupted, resumes from last successful checkpoint
- ✅ Data integrity validated to prevent corruption
- ✅ Failed models tracked and skipped on resume

### **Usage Scenarios**

#### **Scenario 1: First Time Running**
```bash
# Simply run evaluation - checkpointing works automatically
python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml
```

#### **Scenario 2: Evaluation Interrupted**
If your evaluation gets interrupted:

1. **Find your experiment ID** in recent logs:
   ```bash
   grep "Experiment ID" ../../../results/logs/clipcc_eval_*.log
   # Output: Experiment ID: clipcc_eval_20250721_143022
   ```

2. **Resume with the same experiment ID:**
   ```bash
   python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml --experiment_id clipcc_eval_20250721_143022
   ```

#### **Scenario 3: Start Fresh Evaluation**
```bash
# Force fresh start (ignore existing checkpoints)
python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml --experiment_id fresh_evaluation
```

### **What Gets Saved**
- Which models have been completely processed
- Which models failed (to skip them on resume)
- Processing statistics and performance metrics
- Individual results for completed models

## 🛠️ Architecture & Utilities

The framework provides a production-ready architecture optimized for large-scale VLM evaluation:

### `utils/core.py` - **Core Components**
- **ConfigLoader**: Configuration loading and validation
- **DataLoader**: CLIP-CC dataset and VLM prediction loading
  - Supports multiple prediction formats
  - Handles missing data gracefully
  - Validates data consistency
- **VCSEvaluator**: Multi-configuration VCS computation
  - Parallel evaluation across chunk_size and LCT combinations
  - Robust error handling for VCS computation
  - Performance optimization for large datasets
- **ResultsProcessor**: Results formatting and aggregation
  - Individual CSV generation with proper formatting
  - Statistical aggregation with mean ± std
  - Consistent decimal precision across outputs
- **ModelInitializer**: Centralized model initialization (SAT, NV-embed)
- **VLMEvaluationResult**: Data structure for evaluation results

### **Key Features**
- **Thread-Safe Processing**: Parallel model evaluation with race condition prevention
- **Memory Optimization**: Efficient processing without memory accumulation
- **Error Recovery**: Robust handling of model-specific failures
- **Data Validation**: Consistency checks across models and ground truth
- **Performance Monitoring**: Detailed timing and resource usage tracking

## 📊 Data Formats

### **CLIP-CC Ground Truth Format**
```json
[
  {
    "id": "001",
    "file_link": "https://www.youtube.com/watch?v=...",
    "summary": "Detailed human-written description of the video clip..."
  }
]
```

### **VLM Prediction Formats**

**Format 1: List Structure**
```json
[
  {
    "id": "001",
    "summary": "VLM-generated description...",
    "file_link": "https://www.youtube.com/watch?v=..."
  }
]
```

**Format 2: Dictionary Structure (Recommended)**
```json
{
  "001": "VLM-generated description for clip 001...",
  "002": "VLM-generated description for clip 002...",
  "003": "VLM-generated description for clip 003..."
}
```

## 🧪 Reproducing Results

### **Step-by-Step Reproduction Guide**

1. **Environment Setup**
   ```bash
   # Clone and setup VCS
   git clone https://github.com/hdubey-debug/vcs.git
   cd vcs
   pip install -e ".[dev]"
   
   # Navigate to VLM evaluation
   cd vlm_eval
   ```

2. **Download CLIP-CC Dataset**
   ```bash
   python -c "
   from datasets import load_dataset
   import json
   from pathlib import Path
   
   dataset = load_dataset('IVSL-SDSU/Clip-CC')
   output_dir = Path('data/clip-cc')
   output_dir.mkdir(parents=True, exist_ok=True)
   
   data_list = [dict(item) for item in dataset['train']]
   with open(output_dir / 'clip_cc_dataset.json', 'w') as f:
       json.dump(data_list, f, indent=2)
   print(f'Downloaded {len(data_list)} samples')
   "
   ```

3. **Prepare VLM Predictions**
   ```bash
   # Place your VLM prediction JSON files in data/models/
   ls data/models/
   # Should show: internvl.json, llava_next.json, etc.
   ```

4. **Update Configuration**
   ```bash
   # Edit src/config/clipcc_eval_vlms.yaml
   # Update paths to absolute paths for your system
   # Update nv_embed_path to your NV-Embed model location
   ```

5. **Run Evaluation**
   ```bash
   cd src/scripts
   python clipcc_eval_vlms.py --config ../config/clipcc_eval_vlms.yaml
   ```

6. **Collect Results**
   ```bash
   # Individual results per model
   ls ../../results/individual_results/
   
   # Aggregated statistics
   cat ../../results/aggregated_results/aggregated_results.csv
   
   # Processing logs
   tail ../../results/logs/clipcc_eval_*.log
   ```

### **Expected Outputs**
- **Individual CSV files**: One per VLM with detailed scores
- **Aggregated CSV**: Statistical summary across all VLMs
- **Log files**: Detailed processing information
- **Processing time**: ~30-60 minutes for 12 VLMs (depending on hardware)

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd vcs/vlm_eval/src/scripts
   # Verify VCS installation
   python -c "import vcs; print(vcs.__version__)"
   # Check utils
   python -c "from utils.core import ConfigLoader; print('Utils working')"
   ```

2. **Dataset Loading Issues**
   ```bash
   # Verify CLIP-CC dataset
   python -c "
   import json
   with open('../../data/clip-cc/clip_cc_dataset.json') as f:
       data = json.load(f)
   print(f'Loaded {len(data)} samples')
   "
   
   # Check model predictions
   ls ../../data/models/*.json
   ```

3. **Model Path Issues**
   ```bash
   # Update paths in config file
   grep -A5 "nv_embed_path" ../config/clipcc_eval_vlms.yaml
   # Verify NV-Embed model location
   ls /path/to/nv-embed/
   ```

4. **Memory Issues**
   ```bash
   # Reduce max_workers in config
   # Check available RAM
   free -h
   # Monitor GPU memory
   nvidia-smi
   ```

5. **Configuration Errors**
   ```bash
   # Test config loading
   python -c "
   from utils.core import ConfigLoader
   config = ConfigLoader.load_config('../config/clipcc_eval_vlms.yaml')
   print('Config loaded successfully')
   "
   ```

### Performance Optimization

- **Parallel Processing**: Optimal worker allocation based on system resources
  - 2-4 workers: 8GB RAM systems
  - 8-12 workers: 16GB RAM systems  
  - 12+ workers: 32GB+ RAM systems

- **Memory Management**: Optimized for large-scale processing
  - Constant memory footprint regardless of model count
  - Individual result storage prevents memory accumulation
  - Efficient data structures for VCS computation

- **Storage Optimization**:
  - Individual results stored separately for efficiency
  - Automatic cleanup of temporary files
  - Compressed logging for reduced storage

## 🔬 Research Usage

### **Evaluation Metrics**
- **VCS Variants**: 4 configurations per model (chunk_size=[1,2], LCT=[0,1])
- **Statistical Analysis**: Mean ± standard deviation across all video clips
- **Comprehensive Coverage**: All 200 CLIP-CC samples evaluated
- **Reproducible Results**: Consistent preprocessing and evaluation

### **Framework Benefits for Research**
- **Standardized Evaluation**: Consistent VCS computation across all VLMs
- **Multiple Configurations**: Comprehensive analysis with different VCS settings
- **Statistical Rigor**: Proper aggregation with standard deviation reporting
- **Reproducibility**: Detailed logging and configuration management
- **Scalability**: Easy addition of new VLMs to existing evaluation

### **Citation**
If using this VLM evaluation framework in your research, please cite the VCS paper:

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
2. Create a feature branch: `git checkout -b vlm-evaluation-feature`
3. Make changes and test with the VLM evaluation framework
4. Test with multiple VLMs to ensure robustness
5. Submit pull request with detailed description

## 📄 License

This VLM evaluation framework is part of the VCS project and follows the same license terms.

## 🆘 Support

For issues related to:
- **VCS framework**: Open an issue on the [VCS GitHub repository](https://github.com/hdubey-debug/vcs)
- **VLM evaluation**: Check logs in `results/logs/` directory for detailed error information
- **CLIP-CC dataset**: Refer to the [Hugging Face dataset page](https://huggingface.co/datasets/IVSL-SDSU/Clip-CC)
- **Configuration**: Review configuration file comments for parameter explanations

### Quick Support Checklist
1. Check evaluation logs in `results/logs/`
2. Verify configuration paths in `src/config/clipcc_eval_vlms.yaml`
3. Test dataset loading with sample code above
4. Check model prediction formats match expected structure
5. Monitor GPU/RAM usage during evaluation
6. Test with single model first for debugging

---

## 🎯 Framework Summary

This production-ready VLM evaluation framework provides:

### ✅ **Comprehensive VLM Analysis**
- **Multi-Configuration**: 4 VCS variants per model for thorough evaluation
- **Large-Scale**: Supports 10+ VLMs with 200 video clips efficiently  
- **Statistical Rigor**: Proper aggregation with mean ± std reporting

### ✅ **Production Features**
- **Parallel Processing**: Multi-threaded evaluation with intelligent worker management
- **Robust Pipeline**: Error handling, checkpointing, and automatic recovery
- **Memory Efficient**: Constant memory usage regardless of model count
- **Progress Monitoring**: Real-time tracking with detailed analytics

### ✅ **Research-Ready**
- **Reproducible**: Consistent evaluation methodology across all VLMs
- **Standardized**: CLIP-CC dataset integration with proper data handling
- **Comprehensive**: Individual and aggregated results for complete analysis
- **Extensible**: Easy addition of new VLMs and evaluation configurations

### ✅ **Easy Deployment**
- **Single Configuration**: One YAML file controls all evaluation settings
- **Dataset Integration**: Automated CLIP-CC download from Hugging Face
- **Multiple Formats**: Supports various VLM prediction formats
- **Cross-Platform**: Works across different compute environments

**🚀 Ready for comprehensive VLM evaluation on CLIP-CC with standardized VCS metrics and robust statistical analysis.**