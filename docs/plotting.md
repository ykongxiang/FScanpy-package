# Reuse prediction plots

The existing `plot_prf_prediction()` and `PRFPredictor.plot_sequence_prediction()` calls remain valid. Their positional parameters and `(results, figure)` return values are unchanged. Version 1.0.1 adds keyword-only plotting options:

```python
from FScanpy import plot_prf_prediction

results, figure = plot_prf_prediction(
    sequence,
    window_size=3,
    short_threshold=0.2,
    long_threshold=0.2,
    ensemble_weight=0.6,
    reference_positions=[309],
    heatmap_ratios=(0.35, 0.35, 2.8),
    dpi=120,
)
```

`reference_positions` uses independently supplied 0-based nucleotide coordinates, displayed as green dashed lines. The upper candidate heatmap is still generated from model scores; it does not encode reference annotations. The original default panel ratios `(0.1, 0.1, 1)` remain available, while `(0.35, 0.35, 2.8)` gives the thick heatmaps used in the prediction tutorial. `candidate_threshold` controls the candidate-strip cutoff and defaults to 0.8.

Plot an existing table without loading a model or repeating prediction:

```python
from FScanpy import plot_prediction_results, plot_prediction_regions

_, figure = plot_prediction_results(
    results, sequence_length=len(sequence),
    short_threshold=0.2, long_threshold=0.2,
    reference_positions=[309],
    heatmap_ratios=(0.35, 0.35, 2.8), dpi=120,
)

summary, figure = plot_prediction_regions(
    results,
    centers=[("Reference", 309), ("Competitor", 645), ("Competitor", 252)],
    radius=15, short_threshold=0.2, long_threshold=0.2,
    score_threshold=0.7, reference_positions=[309],
)
```

Both helpers preserve the input table. Local plots use equal nucleotide ranges and the same probability scale. The summary contains `region`, `center`, `high_score_positions`, `scanned_positions` and `local_max`. Counts refer to scanned output rows; repeated or overlapping model windows are not independent observations. Empty regions have `local_max=NaN`. Changing display thresholds filters existing scores; it cannot recover long-model scores that were skipped by an earlier inference gate. Recompute with a lower `short_threshold` if needed. The existing predict-and-plot API automatically lowers its inference gate when the short display threshold is below 0.1.

`window_size` controls how many nucleotides to advance between prediction calls, rather than the 33/399 bp model input lengths. A larger scanning interval reduces work for long sequences. Codon-aligned windows remain the same; per-position scans preserve their repeated output rows and reuse their inference.

For a complete runnable example using bundled data, see [the completed plotting notebook](../examples/reusable_plotting.ipynb).

The documented `predictor.extract_features(sequences)` and `predictor.get_model_info()` methods are now implemented. They return a 2D short-model feature array and model types, backend and effective input lengths respectively. The obsolete `SequenceFeatureExtractor.predict_region_batch()` emits a deprecation warning and delegates to the public region predictor; use `PRFPredictor.predict_regions()` directly. Region prediction uses the central 33 bp of `Long_Sequence`/`399bp` as its short-model input.

FScanR alignment coordinates retain the BLASTX 1-based convention. `extract_prf_regions(fasta, sites)` now converts these to 0-based coordinates on the coding strand before extracting windows. If your input table is already 0-based, call `extract_prf_regions(fasta, sites, coordinate_base=0)`. `FS_start` and `FS_end` in the returned table retain the input coordinates. See [the FScanR reference implementation](https://github.com/seanchen607/FScanR/blob/master/R/FScanR.R) and [NCBI BLASTX documentation](https://blast.ncbi.nlm.nih.gov/Blast.cgi?LINK_LOC=blasthome&PAGE_TYPE=BlastSearch&PROGRAM=blastx).
