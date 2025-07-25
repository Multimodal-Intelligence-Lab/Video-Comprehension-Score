# VATEX-EVAL with VCS

A minimalist guide for evaluating VATEX-EVAL dataset using VCS (Video Comprehension Score).

## 📁 Directory Structure

```
vatex-eval/
├── scripts/                        # Evaluation scripts
│   └── vatex_eval_ablation.py     # VCS ablation analysis
├── config/                         # Configuration files
│   └── ablation.yaml              # Ablation experiment config
├── utils/                          # Shared utility modules
│   └── core.py                    # Core classes, constants, and utilities
├── logs/                           # Execution logs
│   └── ablation/                  # Ablation experiment logs
└── README.md                       # This file
```

## Quick Start

### Prerequisites
```bash
pip install vcs torch transformers pandas numpy nltk scipy wtpsplit
```

**NLTK Data Setup:**
```bash
# Standard download (try this first)
python -c "import nltk; nltk.download('stopwords')"

# If SSL certificate errors occur (common in HPC/cluster environments)
python -c "import ssl; import nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('stopwords')"
```

### Basic Usage
```bash
cd vcs/exp/src/vatex-eval
python scripts/vatex_eval_ablation.py --config config/ablation.yaml
```

### Advanced Usage
```bash
# Enable verbose output
python scripts/vatex_eval_ablation.py --config config/ablation.yaml --verbose

# Override experiment ID from config file
python scripts/vatex_eval_ablation.py --config config/ablation.yaml --experiment_id my_custom_id

# Combine options
python scripts/vatex_eval_ablation.py --config config/ablation.yaml --experiment_id vatex_run_001 --verbose
```

## Configuration

Update `config/ablation.yaml` with your paths:

```yaml
models:
  nv_embed_path: "/path/to/nv-embed"

vatex_eval:
  data_dir: "/path/to/vid_cans_score_dict-TEST.json"
  use_n_refs: [1, 9]

paths:
  results_dir: "/path/to/results/vatex-eval"

vcs:
  lct_values: [0]
  chunk_size: 1

processing:
  max_workers: 8
```

## Data Format

Input JSON format:
```json
{
  "video_id": {
    "candidates": ["Generated caption"],
    "references": ["Reference 1", "Reference 2", ...],
    "human_scores": [4.2, 3.8, 4.1]
  }
}
```

## Output

Results saved to `results/vatex-eval/ablation/LCT_0/n_refs_*/`:
- `detailed_scores.csv`: Per-video VCS scores
- `correlation_summary.csv`: Correlation with human scores  
- `ablation_results.csv`: VCS component analysis

## 🛠️ Architecture & Utilities

The framework provides a streamlined architecture with shared utility modules:

### `utils/core.py` - **Core Components**
- **Constants**: Shared constants for consistent processing
  - `PUNCTUATIONS`: Text processing punctuation handling
  - `DECIMAL_PRECISION`: Output formatting (4 decimal places)
  - `ABLATION_METRICS_ORDER`: Standardized ablation metric ordering
  - `DEFAULT_SEGMENTER_FUNCTION`: Default text segmentation approach
- **ConfigLoader**: Configuration loading and validation
- **ModelInitializer**: Centralized model initialization (NV-embed)
- **CheckpointManager**: Production checkpoint system for large-scale processing
- **TextProcessor**: Text preprocessing and segmentation (uses `segmenter_punc_stop` only)
- **EmbeddingGenerator**: Centralized embedding generation
- **VATEXEvalUtils**: VATEX-specific data processing utilities

## VCS Components Evaluated

- **GAS**: Global Alignment Score
- **LAS**: Local Alignment Score  
- **NAS-D**: Narrative Arc Score (Dependency)
- **NAS-L**: Narrative Arc Score (Lexical)
- **VCS**: Complete score

## Troubleshooting

### Common Issues

1. **NLTK Stopwords Error**
   ```bash
   # If you see "Resource stopwords not found", try standard download:
   python -c "import nltk; nltk.download('stopwords')"
   
   # If SSL certificate errors occur, use bypass method:
   python -c "import ssl; import nltk; ssl._create_default_https_context = ssl._create_unverified_context; nltk.download('stopwords')"
   
   # Verify stopwords are available:
   python -c "from nltk.corpus import stopwords; print('Stopwords loaded:', len(stopwords.words('english')))"
   ```

2. **General Verification**
   ```bash
   # Verify installation
   python -c "import vcs; print('VCS installed')"
   
   # Check shared utilities
   python -c "from utils.core import DECIMAL_PRECISION; print('Utils working')"
   
   # Verify constants available
   python -c "from utils.core import ABLATION_METRICS_ORDER, DEFAULT_SEGMENTER_FUNCTION; print('Constants loaded')"
   
   # Test segmenter function
   python -c "from utils.core import TextProcessor; seg = TextProcessor.segmenter_punc_stop; print('Segmenter test:', seg('Hello world test.'))"
   ```

3. **Data and Monitoring**
   ```bash
   # Check data format
   python -c "import json; data=json.load(open('path/to/data.json')); print(f'{len(data)} videos')"
   
   # Monitor progress
   tail -f logs/ablation/vatex_ablation_*.log
   ```