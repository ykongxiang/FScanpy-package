# Changelog

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
