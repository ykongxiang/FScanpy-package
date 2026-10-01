# Changelog

## 1.0.1 (unreleased)

### Fixed
- Preserve scanned positions when both window flanks require N-padding.
- Crop odd length differences to the exact model input length; pad short feature inputs to the trained dimension.
- Propagate model failures instead of silently returning zero probabilities, and prevent input metadata from overwriting computed scores.
- Normalize RNA U and ambiguous bases consistently for both models; compute long-model scores when plotting with a short display threshold below 0.1.
- Implement the documented `extract_features()` and `get_model_info()` methods and route the obsolete feature-extractor prediction helper through the public predictor.
- Preserve the maximum score in overlapping heatmap marks and save PDF siblings without rewriting directory names.
- Deduplicate FScanR peptide sites with peptide coordinates. Convert BLASTX coordinates to 0-based coding-strand positions before extracting windows, including reverse-complement coordinates. Bundled BLASTX results change from 16 to 11 after correct deduplication.

### Added
- `plot_prediction_results()` for existing prediction tables and `plot_prediction_regions()` for side-by-side local comparisons without repeating inference.
- Optional reference markers, heatmap panel ratios and candidate thresholds in existing plotting APIs; existing positional parameters and return structures are retained.
- A completed reusable plotting notebook, bilingual API usage notes and regression tests.
- Bundle the five current prediction-tutorial sequences as `predict_sample_examples.csv`. Updated English and Chinese tutorials load these data through the package API and use the package plotting helpers directly.

### Changed
- Reuse inference for repeated codon windows while preserving every requested scan output row.
- `extract_prf_regions()` defaults to BLASTX 1-based input coordinates; explicitly pass `coordinate_base=0` for 0-based tables. Original coordinate columns remain unchanged in the returned table.

## 1.0.0

### Fixed
- Include pretrained model weights and example data in package distributions.
- Locate bundled models without the deprecated `pkg_resources` API.
- Load PyTorch checkpoints with `weights_only=True` and pin scikit-learn to 1.7.2 for the bundled short model.
- Accept region DataFrames containing `Long_Sequence` or `399bp`.
- Accept pathlib paths when saving prediction plots.
- Export `fscanr` and `extract_prf_regions` from the public package API.
- Repair notebook examples and document Jupyter installation.

### Added
- Package regression tests and a notebook validation runner.
- PyPI project metadata, maintainer contact, MIT license file, and project links.

### Changed
- Align the package's reported version with distribution version 1.0.0.
