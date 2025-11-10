# FScanpy Tutorial - Complete Usage Guide

[![中文](https://img.shields.io/badge/Language-中文-red.svg)](tutorial_zh.md)

## Abstract
FScanpy is a Python package designed to predict Programmed Ribosomal Frameshifting (PRF) sites in DNA sequences. This package integrates machine learning models, sequence feature analysis, and visualization capabilities to help researchers rapidly locate potential PRF sites.

## Introduction
![FScanpy structure](/image/structure.jpeg)

FScanpy is a Python package dedicated to predicting Programmed Ribosomal Frameshifting (PRF) sites in DNA sequences. It integrates machine learning models (Gradient Boosting and BiLSTM-CNN) along with the FScanR package to furnish precise PRF predictions. Users are capable of employing three types of data as input: the entire cDNA/mRNA sequence that requires prediction, the nucleotide sequence in the vicinity of the suspected frameshift site, and the peptide library blastx results of the species or related species. It anticipates the input sequence to be in the + strand and can be integrated with FScanR to augment the accuracy.

![Machine learning models](/image/ML.png)

For the prediction of the entire sequence, FScanpy adopts a sliding window approach to scan the entire sequence and predict the PRF sites. For regional prediction, it is based on the 33-bp and 399-bp sequences in the 0 reading frame around the suspected frameshift site. Initially, the Short model (HistGradientBoosting) will predict the potential PRF sites within the scanning window. If the predicted probability exceeds the threshold, the Long model (BiLSTM-CNN) will predict the PRF sites in the 399bp sequence. Then, ensemble weighting combines the two models to make the final prediction.

For PRF detection from BLASTX output, [FScanR](https://github.com/seanchen607/FScanR.git) identifies potential PRF sites from BLASTX alignment results, acquires the two hits of the same query sequence, and then utilizes frameDist_cutoff, mismatch_cutoff, and evalue_cutoff to filter the hits. Finally, FScanpy is utilized to predict the probability of PRF sites.

### Background
[Ribosomal frameshifting](https://en.wikipedia.org/wiki/Ribosomal_frameshift), also known as translational frameshifting or translational recoding, is a biological phenomenon that occurs during translation that results in the production of multiple, unique proteins from a single mRNA. The process can be programmed by the nucleotide sequence of the mRNA and is sometimes affected by the secondary, 3-dimensional mRNA structure. It has been described mainly in viruses (especially retroviruses), retrotransposons and bacterial insertion elements, and also in some cellular genes.

### Key features of FScanpy include:

- Integration of two predictive models:
  - Short Model (HistGradientBoosting): Analyzes local sequence features centered around potential frameshift sites (33bp).
  - Long Model (BiLSTM-CNN): Analyzes broader sequence features (399bp).
- Supports PRF prediction across various species.
- Can be combined with [FScanR](https://github.com/seanchen607/FScanR.git) for enhanced accuracy.

## Installation (python>=3.11)

### 1. Use pip
```bash
pip install FScanpy
```

### 2. Clone from GitHub
```bash
git clone https://github.com/.../FScanpy.git
cd FScanpy
pip install -e .
```

## Complete Function Reference

### 1. Core Prediction Functions

#### 1.1 `predict_prf()` - Main Prediction Interface

**Function Signature:**
```python
def predict_prf(
    sequence: Union[str, List[str], None] = None,
    data: Union[pd.DataFrame, None] = None,
    window_size: int = 3,
    short_threshold: float = 0.1,
    ensemble_weight: float = 0.4,
    model_dir: str = None
) -> pd.DataFrame
```

**Parameters:**
- `sequence`: Single or multiple DNA sequences for sliding window prediction
- `data`: DataFrame data, must contain 'Long_Sequence' or '399bp' column for region prediction  
- `window_size`: Sliding window size (default: 3, recommended: 1-10)
- `short_threshold`: Short model (HistGB) probability threshold (default: 0.1, range: 0.0-1.0)
- `ensemble_weight`: Weight of short model in ensemble (default: 0.4, range: 0.0-1.0)
- `model_dir`: Model directory path (optional, uses built-in models if None)

**Returns:**
- `pd.DataFrame`: Prediction results with columns:
  - `Short_Probability`: Short model prediction probability
  - `Long_Probability`: Long model prediction probability  
  - `Ensemble_Probability`: Ensemble prediction probability (main result)
  - `Position`: Position in sequence (for sliding window mode)
  - `Codon`: Codon at position (for sliding window mode)
  - `Ensemble_Weights`: Weight configuration information

**Usage Examples:**

```python
from FScanpy import predict_prf

# 1. Single sequence sliding window prediction
sequence = "ATGCGTACGTATGCGTACGTATGCGTACGT"
results = predict_prf(sequence=sequence)

# 2. Multiple sequences prediction
sequences = ["ATGCGTACGT...", "GCTATAGCAT..."]
results = predict_prf(sequence=sequences)

# 3. Custom parameters
results = predict_prf(
    sequence=sequence, 
    window_size=1,           # Scan every position
    short_threshold=0.2,     # Higher threshold
    ensemble_weight=0.3      # 3:7 ratio (short:long)
)

# 4. DataFrame region prediction
import pandas as pd
data = pd.DataFrame({
    'Long_Sequence': ['ATGCGT...', 'GCTATAG...'],  # or use '399bp'
    'sample_id': ['sample1', 'sample2']
})
results = predict_prf(data=data)
```

#### 1.2 `plot_prf_prediction()` - Prediction with Visualization

**Function Signature:**
```python
def plot_prf_prediction(
    sequence: str,
    window_size: int = 3,
    short_threshold: float = 0.65,
    long_threshold: float = 0.8,
    ensemble_weight: float = 0.4,
    title: str = None,
    save_path: str = None,
    figsize: tuple = (12, 8),
    dpi: int = 300,
    model_dir: str = None
) -> tuple
```

**Parameters:**
- `sequence`: Input DNA sequence (string)
- `window_size`: Sliding window size (default: 3)
- `short_threshold`: Short model filtering threshold for heatmap display (default: 0.65)
- `long_threshold`: Long model filtering threshold for heatmap display (default: 0.8)
- `ensemble_weight`: Weight of short model in ensemble (default: 0.4)
- `title`: Plot title (optional, auto-generated if None)
- `save_path`: Save path (optional, saves plot if provided)
- `figsize`: Figure size tuple (default: (12, 8))
- `dpi`: Figure resolution (default: 300)
- `model_dir`: Model directory path (optional)

**Returns:**
- `tuple`: (prediction_results: pd.DataFrame, figure: matplotlib.figure.Figure)

**Usage Examples:**

```python
from FScanpy import plot_prf_prediction
import matplotlib.pyplot as plt

# 1. Basic plotting
sequence = "ATGCGTACGTATGCGTACGTATGCGTACGT"
results, fig = plot_prf_prediction(sequence)
plt.show()

# 2. Custom thresholds and weights
results, fig = plot_prf_prediction(
    sequence, 
    short_threshold=0.7,     # Higher display threshold
    long_threshold=0.85,     # Higher display threshold
    ensemble_weight=0.3,     # 3:7 weight ratio
    title="Custom Analysis Results",
    save_path="analysis.png",
    figsize=(15, 10),
    dpi=150
)

# 3. High-resolution analysis
results, fig = plot_prf_prediction(
    sequence,
    window_size=1,           # Scan every position
    ensemble_weight=0.5,     # Equal weights
    dpi=600                  # High resolution
)
```

### 2. PRFPredictor Class Methods

#### 2.1 Class Initialization

```python
from FScanpy import PRFPredictor

# Initialize with default models
predictor = PRFPredictor()

# Initialize with custom model directory
predictor = PRFPredictor(model_dir='/path/to/models')
```

#### 2.2 `predict_sequence()` - Sliding Window Prediction

**Method Signature:**
```python
def predict_sequence(self, sequence, window_size=3, short_threshold=0.1, ensemble_weight=0.4)
```

**Parameters:**
- `sequence`: Input DNA sequence
- `window_size`: Sliding window size (default: 3)
- `short_threshold`: Short model probability threshold (default: 0.1)
- `ensemble_weight`: Short model weight in ensemble (default: 0.4)

**Usage:**
```python
predictor = PRFPredictor()
results = predictor.predict_sequence(
    sequence="ATGCGTACGT...",
    window_size=1,
    short_threshold=0.15,
    ensemble_weight=0.35
)
```

#### 2.3 `predict_regions()` - Region-based Prediction

**Method Signature:**
```python
def predict_regions(self, sequences, short_threshold=0.1, ensemble_weight=0.4)
```

**Parameters:**
- `sequences`: List or Series of 399bp sequences
- `short_threshold`: Short model probability threshold (default: 0.1)
- `ensemble_weight`: Short model weight in ensemble (default: 0.4)

**Usage:**
```python
predictor = PRFPredictor()
sequences = ["ATGCGT...", "GCTATAG..."]  # 399bp sequences
results = predictor.predict_regions(
    sequences=sequences,
    short_threshold=0.1,
    ensemble_weight=0.4
)
```

#### 2.4 `predict_single_position()` - Single Position Prediction

**Method Signature:**
```python
def predict_single_position(self, fs_period, full_seq, short_threshold=0.1, ensemble_weight=0.4)
```

**Parameters:**
- `fs_period`: 33bp sequence around frameshift site
- `full_seq`: 399bp sequence for long model
- `short_threshold`: Short model probability threshold (default: 0.1)
- `ensemble_weight`: Short model weight in ensemble (default: 0.4)

**Usage:**
```python
predictor = PRFPredictor()
result = predictor.predict_single_position(
    fs_period="ATGCGTACGTATGCGTACGTATGCGTACGTA",  # 33bp
    full_seq="ATGCGT..." * 133,  # 399bp
    short_threshold=0.1,
    ensemble_weight=0.4
)
```

#### 2.5 `plot_sequence_prediction()` - Class Method for Plotting

**Method Signature:**
```python
def plot_sequence_prediction(self, sequence, window_size=3, short_threshold=0.65, 
                           long_threshold=0.8, ensemble_weight=0.4, title=None, 
                           save_path=None, figsize=(12, 8), dpi=300)
```

**Usage:**
```python
predictor = PRFPredictor()
results, fig = predictor.plot_sequence_prediction(
    sequence="ATGCGTACGT...",
    window_size=3,
    ensemble_weight=0.4
)
```

### 3. Utility Functions

#### 3.1 `fscanr()` - PRF Site Detection from BLASTX

**Function Signature:**
```python
def fscanr(
    blastx_output: pd.DataFrame,
    mismatch_cutoff: float = 10,
    evalue_cutoff: float = 1e-5,
    frameDist_cutoff: float = 10
) -> pd.DataFrame
```

**Parameters:**
- `blastx_output`: BLASTX output DataFrame with required columns:
  - `qseqid`, `sseqid`, `pident`, `length`, `mismatch`, `gapopen`
  - `qstart`, `qend`, `sstart`, `send`, `evalue`, `bitscore`, `qframe`, `sframe`
- `mismatch_cutoff`: Maximum allowed mismatches (default: 10)
- `evalue_cutoff`: E-value threshold (default: 1e-5)
- `frameDist_cutoff`: Frame distance threshold (default: 10)

**Returns:**
- `pd.DataFrame`: PRF sites with columns:
  - `DNA_seqid`: Sequence identifier
  - `FS_start`, `FS_end`: Frameshift start and end positions
  - `Pep_seqid`: Peptide sequence identifier
  - `Pep_FS_start`, `Pep_FS_end`: Peptide frameshift positions
  - `FS_type`: Type of frameshift (-2, -1, 1, 2)
  - `Strand`: Strand orientation (+, -)

**Usage:**
```python
from FScanpy.utils import fscanr
import pandas as pd

# Load BLASTX results
blastx_data = pd.read_excel('blastx_results.xlsx')

# Detect PRF sites
prf_sites = fscanr(
    blastx_output=blastx_data,
    mismatch_cutoff=5,       # Stricter mismatch filter
    evalue_cutoff=1e-6,      # Stricter E-value filter
    frameDist_cutoff=15      # Allow larger frame distances
)
```

#### 3.2 `extract_prf_regions()` - Extract Sequences Around PRF Sites

**Function Signature:**
```python
def extract_prf_regions(mrna_file: str, prf_data: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `mrna_file`: Path to mRNA sequences file (FASTA format)
- `prf_data`: DataFrame from `fscanr()` output

**Returns:**
- `pd.DataFrame`: Extracted sequences with columns:
  - `DNA_seqid`: Sequence identifier
  - `FS_start`, `FS_end`: Frameshift positions
  - `Strand`: Strand orientation
  - `399bp`: Extracted 399bp sequence
  - `FS_type`: Frameshift type

**Usage:**
```python
from FScanpy.utils import extract_prf_regions

# Extract sequences around PRF sites
prf_sequences = extract_prf_regions(
    mrna_file='sequences.fasta',
    prf_data=prf_sites
)

# Predict PRF probabilities
predictor = PRFPredictor()
results = predictor.predict_regions(prf_sequences['399bp'])
```

### 4. Data Access Functions

#### 4.1 Test Data Access

```python
from FScanpy.data import get_test_data_path, list_test_data

# List available test data
list_test_data()

# Get test data paths
blastx_file = get_test_data_path('blastx_example.xlsx')
mrna_file = get_test_data_path('mrna_example.fasta')
region_file = get_test_data_path('region_example.csv')
seq_file = get_test_data_path('full_seq.xlsx')
```

## Complete Workflow Examples

### Workflow 1: Full Sequence Analysis

```python
from FScanpy import predict_prf, plot_prf_prediction
import matplotlib.pyplot as plt

# Define sequence
full_seq = pd.read.excel(seq_file)

# Method 1: Simple prediction
results = predict_prf(sequence=full_seq[0]['full_seq'])
print(f"Found {len(results)} potential sites")

# Method 2: Prediction with visualization
results, fig = plot_prf_prediction(
    sequence=sequence=full_seq[0]['full_seq'],
    window_size=1,              # Scan every position
    short_threshold=0.3,        # Display sites above 0.3
    long_threshold=0.4,         # Display sites above 0.4
    ensemble_weight=0.4,        # 4:6 weight ratio
    title="PRF Analysis Results",
    save_path="prf_analysis.png"
)
plt.show()

# Analyze top predictions
top_sites = results.nlargest(5, 'Ensemble_Probability')
print("Top 5 predicted sites:")
for _, site in top_sites.iterrows():
    print(f"Position {site['Position']}: {site['Ensemble_Probability']:.3f}")
```

### Workflow 2: Region-based Prediction

```python
from FScanpy import predict_prf
import pandas as pd

# Prepare region data
region_data = pd.DataFrame({
    'sample_id': ['sample1', 'sample2', 'sample3'],
    'Long_Sequence': [
        'ATGCGT...',  # 399bp sequence 1
        'GCTATAG...',  # 399bp sequence 2  
        'TTACGGA...'   # 399bp sequence 3
    ],
    'known_label': [1, 0, 1]  # Optional: known labels for validation
})

# Predict PRF probabilities
results = predict_prf(
    data=region_data,
    ensemble_weight=0.3  # Favor long model (3:7 ratio)
)

# Evaluate results
if 'known_label' in results.columns:
    threshold = 0.5
    predictions = (results['Ensemble_Probability'] > threshold).astype(int)
    accuracy = (predictions == results['known_label']).mean()
    print(f"Accuracy at threshold {threshold}: {accuracy:.3f}")
```

### Workflow 3: BLASTX-based Analysis Pipeline

```python
from FScanpy import PRFPredictor, predict_prf
from FScanpy.data import get_test_data_path
from FScanpy.utils import fscanr, extract_prf_regions
import pandas as pd

# Step 1: Load BLASTX data
blastx_data = pd.read_excel(get_test_data_path('blastx_example.xlsx'))
print(f"Loaded {len(blastx_data)} BLASTX hits")

# Step 2: Detect PRF sites using FScanR
prf_sites = fscanr(
    blastx_output=blastx_data,
    mismatch_cutoff=10,
    evalue_cutoff=1e-5,
    frameDist_cutoff=10
)
print(f"Detected {len(prf_sites)} potential PRF sites")

# Step 3: Extract sequences around PRF sites
mrna_file = get_test_data_path('mrna_example.fasta')
prf_sequences = extract_prf_regions(
    mrna_file=mrna_file,
    prf_data=prf_sites
)
print(f"Extracted {len(prf_sequences)} sequences")

# Step 4: Predict PRF probabilities
predictor = PRFPredictor()
results = predictor.predict_regions(
    sequences=prf_sequences['399bp'],
    ensemble_weight=0.4
)

# Step 5: Combine results with metadata
final_results = pd.concat([
    prf_sequences.reset_index(drop=True),
    results.reset_index(drop=True)
], axis=1)

# Step 6: Analyze results
high_prob_sites = final_results[
    final_results['Ensemble_Probability'] > 0.7
]
print(f"High probability PRF sites: {len(high_prob_sites)}")

# Display top results
print("\nTop PRF predictions:")
top_results = final_results.nlargest(3, 'Ensemble_Probability')
for _, row in top_results.iterrows():
    print(f"Sequence {row['DNA_seqid']}: {row['Ensemble_Probability']:.3f}")
```

### Workflow 4: Custom Analysis with Multiple Sequences

```python
from FScanpy import predict_prf, plot_prf_prediction
import matplotlib.pyplot as plt

# Multiple sequence analysis
sequences = [
    "ATGCGTACGTATGCGTACGTATGCGTACGTAAGCCCTTTGAACCCAAAGGG",
    "GCTATAGCATGCTATAGCATGCTATAGCATGCTATAGCATGCTATAGCAT",
    "TTACGGATTACGGATTACGGATTACGGATTACGGATTACGGATTACGGAT"
]

# Batch prediction
results = predict_prf(
    sequence=sequences,
    window_size=2,
    ensemble_weight=0.5  # Equal weights
)

# Analyze per sequence
for seq_id in results['Sequence_ID'].unique():
    seq_results = results[results['Sequence_ID'] == seq_id]
    max_prob = seq_results['Ensemble_Probability'].max()
    print(f"{seq_id}: Max probability = {max_prob:.3f}")

# Visualize first sequence
first_seq_results, fig = plot_prf_prediction(
    sequence=sequences[0],
    ensemble_weight=0.5,
    title="First Sequence Analysis"
)
plt.show()
```

## Parameter Optimization Guidelines

### 1. Ensemble Weight Selection

- **Conservative (favor specificity)**: `ensemble_weight = 0.2-0.3` (favor long model)
- **Balanced**: `ensemble_weight = 0.4-0.6` (default recommended)
- **Sensitive (favor sensitivity)**: `ensemble_weight = 0.7-0.8` (favor short model)

### 2. Threshold Selection

- **Short threshold**: Usually 0.1-0.3, controls computational efficiency
- **Display thresholds**: 0.3-0.8, controls visualization display
- **Classification threshold**: 0.5 (standard), adjust based on validation data

### 3. Window Size Selection

- **Fine-grained analysis**: `window_size = 1` (every position)
- **Standard analysis**: `window_size = 3` (every 3rd position, default)
- **Coarse analysis**: `window_size = 6-9` (faster, less detailed)

## Troubleshooting

### Common Issues and Solutions

1. **Model Loading Errors**
   ```python
   # Check model directory
   import FScanpy
   predictor = PRFPredictor(model_dir='/custom/path')
   ```

2. **Memory Issues with Large Sequences**
   ```python
   # Use larger window size to reduce computational load
   results = predict_prf(sequence=large_seq, window_size=9)
   ```

3. **Visualization Issues**
   ```python
   # Adjust figure parameters
   results, fig = plot_prf_prediction(
       sequence=seq,
       figsize=(20, 10),  # Larger figure
       dpi=150            # Lower resolution
   )
   ```

4. **Input Format Issues**
   ```python
   # Ensure proper DataFrame format
   data = pd.DataFrame({
       'Long_Sequence': sequences,  # Use 'Long_Sequence' or '399bp'
       'sample_id': ids
   })
   ```

## Performance Optimization

### 1. Batch Processing
```python
# Process multiple sequences efficiently
sequences = ["seq1", "seq2", "seq3", ...]
results = predict_prf(sequence=sequences, window_size=3)
```

### 2. Threshold Optimization
```python
# Use appropriate short_threshold to skip unnecessary long model calls
results = predict_prf(
    sequence=sequence,
    short_threshold=0.2  # Higher threshold = faster processing
)
```

### 3. Memory Management
```python
# For very large datasets, process in chunks
chunk_size = 100
for i in range(0, len(large_dataset), chunk_size):
    chunk = large_dataset[i:i+chunk_size]
    chunk_results = predict_prf(data=chunk)
    # Process chunk_results
```

## Citation
If you use FScanpy, please cite our paper: [Paper Link]