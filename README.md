# FScanpy
## A Machine Learning-Based Framework for Programmed Ribosomal Frameshifting Prediction

[![中文](https://img.shields.io/badge/Language-中文-red.svg)](https://github.com/ykongxiang/FScanpy-package/blob/master/README_zh.md)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/ykongxiang/FScanpy-package/blob/master/LICENSE)

FScanpy is a comprehensive Python package designed for the prediction of [Programmed Ribosomal Frameshifting (PRF)](https://en.wikipedia.org/wiki/Ribosomal_frameshift) sites in nucleotide sequences. By integrating advanced machine learning approaches (HistGradientBoosting and BiLSTM-CNN) with the established [FScanR](https://github.com/seanchen607/FScanR.git) framework, FScanpy provides robust and accurate PRF site predictions.

![FScanpy Architecture](https://raw.githubusercontent.com/ykongxiang/FScanpy-package/master/tutorial/image/structure.jpg)

## 🔧 Installation

### Prerequisites
- Python ≥ 3.11
- All dependencies are automatically installed

### Install via pip (Recommended)
```bash
pip install FScanpy
```

### Install from Source
```bash
git clone https://github.com/ykongxiang/FScanpy-package.git
cd FScanpy-package
pip install -e .
```

### Jupyter Notebook

From the cloned package directory, install into the Python environment you will use for Jupyter:

```bash
python -m pip install . notebook ipykernel
python -m ipykernel install --sys-prefix --name fscanpy --display-name "Python (FScanpy)"
python -m notebook
```

Open `FScanpy_Demo.ipynb` or `tutorial/predict_sample.ipynb`, select **Python (FScanpy)**, then restart the kernel and run all cells. The API overview in the demo is explanatory Markdown; the runnable examples use the bundled data. If installing into an already open notebook, use `%pip install /path/to/FScanpy-package` and restart the kernel.

## 🚀 Quick Start

### Basic Usage
```python
from FScanpy import predict_prf

# Simple sequence prediction
sequence = "ATGCGTACGTTAGC"*100 # Your DNA sequence
results = predict_prf(sequence=sequence)

# View top predictions
print(results[['Position', 'Ensemble_Probability', 'Short_Probability', 'Long_Probability']].head(10))
```

### Visualization
```python
from FScanpy import plot_prf_prediction

# Generate prediction plot
results, fig = plot_prf_prediction(
    sequence=sequence,
    short_threshold=0.65,    # HistGB threshold
    long_threshold=0.8,      # BiLSTM-CNN threshold
    ensemble_weight=0.4,     # 40% Short, 60% Long
    title="PRF Prediction Results"
)
```

### Advanced Usage
```python
from FScanpy import PRFPredictor
import pandas as pd

# Create predictor instance
predictor = PRFPredictor()

# Batch prediction on pre-extracted regions
data = pd.DataFrame({
    'Long_Sequence': ['ATG' * 133, 'GCT' * 133]  # 399bp sequences
})
results = predictor.predict_regions(data, ensemble_weight=0.4)

# Sequence-level prediction with custom parameters
results = predictor.predict_sequence(
    sequence=sequence,
    window_size=1,           # Step size for sliding window
    ensemble_weight=0.3,     # Model weighting
    short_threshold=0.5      # Filtering threshold
)
```

## 🎛️ Ensemble Weight Configuration

The `ensemble_weight` parameter controls the weight ratio between HistGB and BiLSTM-CNN models:

| ensemble_weight | HistGB Model | BiLSTM-CNN Model | Characteristics | Best For |
|----------------|-------------|------------------|-----------------|----------|
| **0.2-0.3** | 20-30% | 70-80% | **High specificity**, reduces false positives | Precise validation, clinical applications |
| **0.4** | 40% | 60% | **Optimal balance**, highest AUC | Standard analysis (recommended) |
| **0.6-0.8** | 60-80% | 20-40% | **High sensitivity**, captures more sites | High-throughput screening, exploratory research |


### Weight Selection Examples
```python
# High specificity configuration (favoring HistGB)
precise_results = predict_prf(sequence, ensemble_weight=0.25)

# Optimal balance configuration (4:6 ratio)
balanced_results = predict_prf(sequence, ensemble_weight=0.4)

# High sensitivity configuration (favoring BiLSTM-CNN)
sensitive_results = predict_prf(sequence, ensemble_weight=0.7)
```

## 📊 Core Functions

### Main Prediction Interface
```python
predict_prf(
    sequence=None,           # Single/multiple sequences or None
    data=None,              # DataFrame with 399bp sequences or None
    window_size=3,          # Sliding window step size
    short_threshold=0.1,    # Short model filtering threshold
    ensemble_weight=0.4,    # Short model weight (0.0-1.0)
    model_dir=None         # Custom model directory
)
```

### Visualization Function
```python
plot_prf_prediction(
    sequence,               # Input DNA sequence
    window_size=3,          # Scanning step size
    short_threshold=0.65,   # Short model threshold for plotting
    long_threshold=0.8,     # Long model threshold for plotting
    ensemble_weight=0.4,    # Model weighting
    title=None,            # Plot title
    save_path=None,        # Save file path
    figsize=(12,8),        # Figure size
    dpi=300               # Resolution for saved plots
)
```

### PRFPredictor Class Methods
```python
predictor = PRFPredictor()

# Sequence prediction (sliding window)
predictor.predict_sequence(sequence, ensemble_weight=0.4)

# Region prediction (batch processing)
predictor.predict_regions(dataframe, ensemble_weight=0.4)

# Feature extraction
predictor.extract_features(sequences)

# Model information
predictor.get_model_info()
```

## 📈 Output Fields

### Prediction Results
- **`Position`**: Position in the original sequence
- **`Ensemble_Probability`**: Final ensemble prediction (main result)
- **`Short_Probability`**: HistGradientBoosting prediction (0-1)
- **`Long_Probability`**: BiLSTM-CNN prediction (0-1)
- **`Ensemble_Weights`**: Model weight configuration used

### Sequence Information
- **`Short_Sequence`**: 33bp sequence for Short model
- **`Long_Sequence`**: 399bp sequence for Long model  
- **`Codon`**: 3bp codon at the prediction position
- **`Sequence_ID`**: Identifier for multi-sequence inputs

## 🔬 Integration with FScanR

FScanpy works seamlessly with the FScanR pipeline for comprehensive PRF analysis:

```python
from FScanpy import fscanr, extract_prf_regions, predict_prf

# Step 1: BLASTX analysis with FScanR
blastx_results = fscanr(
    blastx_data,
    mismatch_cutoff=10,
    evalue_cutoff=1e-5,
    frameDist_cutoff=10
)

# Step 2: Extract PRF candidate regions
prf_regions = extract_prf_regions(original_sequence, blastx_results)

# Step 3: Predict with FScanpy
final_predictions = predict_prf(data=prf_regions, ensemble_weight=0.4)
```

## 📚 Documentation

- **[Complete Tutorial](https://github.com/ykongxiang/FScanpy-package/blob/master/tutorial/tutorial.md)**: Comprehensive usage guide with examples
- **[Demo Notebook](https://github.com/ykongxiang/FScanpy-package/blob/master/FScanpy_Demo.ipynb)**: Practical usage of each function in the library and demonstration of analysis workflow results
- **[Predict Sample Interpretation](https://github.com/ykongxiang/FScanpy-package/blob/master/tutorial/predict_sample.ipynb)**: Detailed interpretation of FScanpy's plotting results and signal analysis

## 📝 Citation

If you use FScanpy in your research, please cite:

```bibtex
@article{yang2026deciphering,

  author    = {Yang, Yu-Hao and Yang, Juan and Liu, Zi-Jia and Li, Yuan and Song, Weibo and Stover, Naomi and Chen, Xiao},

  title     = {Deciphering ribosomal frameshifting determinants across species with a semi-supervised hybrid learning framework},

  journal   = {Zoological Research},

  doi       = {10.24272/j.issn.2095-8137.2025.648},

  url       = {https://doi.org/10.24272/j.issn.2095-8137.2025.648}

}
```


**FScanpy** - Advancing programmed ribosomal frameshifting research through machine learning 🧬

## Maintainer and License

Primary maintainer: **Yang Yuhao** ([ykongxiang@qq.com](mailto:ykongxiang@qq.com)).

FScanpy is distributed under the [MIT License](https://github.com/ykongxiang/FScanpy-package/blob/master/LICENSE). See the [changelog](https://github.com/ykongxiang/FScanpy-package/blob/master/CHANGELOG.md) for version 1.0.0 changes.
