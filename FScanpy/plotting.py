"""Plot saved predictions without loading a model or repeating inference."""
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._validation import positive_integer, probability

PROBABILITY_COLUMNS = ['Short_Probability', 'Long_Probability', 'Ensemble_Probability']


def _table(results):
    if not isinstance(results, pd.DataFrame):
        raise ValueError('results must be a prediction DataFrame for one sequence')
    required = ['Position'] + PROBABILITY_COLUMNS
    missing = [name for name in required if name not in results.columns]
    if missing:
        raise ValueError(f'Missing prediction columns: {missing}')
    if results.empty:
        raise ValueError('Prediction results are empty')
    positions = results.Position.to_numpy(dtype=float)
    scores = results[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(positions).all() or (positions < 0).any() or (positions != np.floor(positions)).any():
        raise ValueError('Position must contain nonnegative integer nucleotide coordinates')
    if results.Position.duplicated().any():
        raise ValueError('Position must be unique; pass results for one sequence at a time')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('Prediction probabilities must be finite and between 0 and 1')
    return positions.astype(int)


def _scores(results, short_threshold, long_threshold):
    short_threshold = probability(short_threshold, 'short_threshold')
    long_threshold = probability(long_threshold, 'long_threshold')
    mask = (results.Short_Probability >= short_threshold) & (results.Long_Probability >= long_threshold)
    return results.Ensemble_Probability.where(mask, 0).to_numpy(dtype=float), mask.to_numpy()


def _ratios(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('heatmap_ratios must contain three positive finite numbers')
    return values.tolist()


def _references(values, length=None):
    if values is None:
        return []
    if np.isscalar(values):
        values = [values]
    references = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0 or (length is not None and value >= length):
            raise ValueError('reference_positions must contain 0-based integer coordinates within the sequence')
        references.append(int(value))
    return references


def _save(figure, save_path, dpi):
    if save_path is not None:
        path = Path(save_path)
        figure.savefig(path, dpi=dpi, bbox_inches='tight')
        if path.suffix.lower() == '.png':
            figure.savefig(path.with_suffix('.pdf'), bbox_inches='tight')


def plot_prediction_results(results, sequence_length=None, short_threshold=0.65,
                            long_threshold=0.8, title=None, save_path=None,
                            figsize=(12, 8), dpi=300, *, reference_positions=None,
                            heatmap_ratios=(0.1, 0.1, 1), candidate_threshold=0.8):
    """Plot an existing single-sequence prediction table; return ``(results, figure)``.

    The original two red heatmaps and black bars are retained. Use
    ``heatmap_ratios=(0.35, 0.35, 2.8)`` for the tutorial's thick heatmaps.
    ``reference_positions`` marks independently supplied 0-based positions in
    green; candidate heatmaps remain generated from the prediction scores.
    Overlapping heatmap marks show their maximum score, not the last row's score.
    ``sequence_length`` defaults to the last scanned position plus three; pass
    the actual length to retain the full extent of a sparsely scanned sequence.
    The input table is not modified and no models are loaded.
    """
    positions = _table(results)
    length = positive_integer(sequence_length if sequence_length is not None else int(positions.max()) + 3,
                              'sequence_length')
    if positions.max() >= length:
        raise ValueError('sequence_length must be larger than every scanned Position')
    scores, eligible = _scores(results, short_threshold, long_threshold)
    candidate_threshold = probability(candidate_threshold, 'candidate_threshold')
    ratios = _ratios(heatmap_ratios)
    references = _references(reference_positions, length)

    desired_width = max(3, length // 100)
    probability_width = max(1, desired_width // 3)
    candidates = np.zeros((1, length))
    heatmap_scores = np.zeros((1, length))
    for position, score, visible in zip(positions, scores, eligible):
        if not visible:
            continue
        start, end = max(0, position - probability_width // 2), min(length, position + probability_width // 2 + 1)
        heatmap_scores[0, start:end] = np.maximum(heatmap_scores[0, start:end], score)
        if score >= candidate_threshold:
            start, end = max(0, position - desired_width // 2), min(length, position + desired_width // 2 + 1)
            candidates[0, start:end] = 1

    figure = plt.figure(figsize=figsize, dpi=dpi)
    figure.suptitle(title or 'PRF Prediction Results', fontsize=10)
    grid = figure.add_gridspec(3, 1, height_ratios=ratios)
    axes = [figure.add_subplot(grid[row]) for row in range(3)]
    for axis, data, label in zip(axes[:2], [candidates, heatmap_scores], ['FS site (predicted candidates)', 'Prediction']):
        axis.imshow(data, cmap='Reds', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
        axis.set(xticks=[], yticks=[], title=label)
        axis.title.set_fontsize(10)
    bars = axes[2]
    bars.bar(positions, scores, alpha=0.6, color='black', width=1)
    bars.set(xlabel='Position (0-based nt)', ylabel='Filtered ensemble score', ylim=(0, 1))
    bars.set_xticks(np.arange(0, length, max(length // 10, 50)))
    bars.tick_params(axis='x', rotation=45)
    bars.grid(True, alpha=0.3)
    for axis in axes:
        axis.set_xlim(-1, length)
        for reference in references:
            axis.axvline(reference, color='#009E73', linestyle='--', linewidth=1.2,
                         label=f'Reference {reference}' if axis is bars else None)
    if references:
        bars.legend(loc='upper right', fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    _save(figure, save_path, dpi)
    return results, figure


def plot_prediction_regions(results, centers, radius=15, short_threshold=0.2,
                            long_threshold=0.2, score_threshold=0.7, *,
                            candidate_threshold=0.8, reference_positions=None,
                            heatmap_ratios=(0.35, 0.35, 2.8), figsize=None,
                            dpi=120, save_path=None):
    """Compare equally sized regions of existing predictions, without inference.

    ``centers`` is a list of 0-based positions or ``(label, position)`` pairs,
    e.g. ``[('Reference', 309), ('Competitor', 645)]``. Return ``(summary, figure)``.
    Each column contains two thick heatmaps above bars with the same 0–1 scale.
    ``high_score_positions`` counts scanned rows reaching ``score_threshold``;
    nearby scores can share overlapping input windows and are not independent
    observations. Regions with no scanned rows have NaN ``local_max``.
    """
    positions = _table(results)
    scores, eligible = _scores(results, short_threshold, long_threshold)
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0:
        raise ValueError('radius must be a nonnegative integer in nucleotides')
    radius = int(radius)
    score_threshold = probability(score_threshold, 'score_threshold')
    candidate_threshold = probability(candidate_threshold, 'candidate_threshold')
    ratios = _ratios(heatmap_ratios)
    references = _references(reference_positions)
    regions = []
    for item in centers:
        if isinstance(item, Integral) and not isinstance(item, bool):
            label, center = 'Region', item
        else:
            label, center = item
        if isinstance(center, bool) or not isinstance(center, Integral) or center < 0:
            raise ValueError('centers must contain nonnegative integer nucleotide coordinates')
        regions.append((str(label), int(center)))
    if not regions:
        raise ValueError('Provide at least one region center')

    figure = plt.figure(figsize=figsize or (max(5, 3.5 * len(regions)), 5.3), dpi=dpi)
    grid = figure.add_gridspec(3, len(regions), height_ratios=ratios)
    summaries = []
    for column, (label, center) in enumerate(regions):
        local = np.abs(positions - center) <= radius
        count = int(np.sum(local & eligible & (scores >= score_threshold)))
        summaries.append(dict(region=label, center=center, high_score_positions=count,
                              scanned_positions=int(local.sum()),
                              local_max=float(scores[local].max()) if local.any() else float('nan')))
        width = 2 * radius + 1
        candidates, heatmap_scores = np.zeros((1, width)), np.zeros((1, width))
        for position, score, visible in zip(positions[local], scores[local], eligible[local]):
            if not visible:
                continue
            index = int(position - center + radius)
            heatmap_scores[0, index] = score
            if score >= candidate_threshold:
                candidates[0, max(0, index - 1):min(width, index + 2)] = 1
        axes = [figure.add_subplot(grid[row, column]) for row in range(3)]
        extent = (-radius - 0.5, radius + 0.5, 0, 1)
        for axis, data in zip(axes[:2], [candidates, heatmap_scores]):
            axis.imshow(data, cmap='Reds', aspect='auto', vmin=0, vmax=1,
                        interpolation='nearest', extent=extent)
            axis.set(xticks=[], yticks=[])
        axes[0].set_title(f'{label}: {center}\nFS site (predicted candidates)', fontsize=9)
        axes[1].set_title('Prediction', fontsize=9)
        bars = axes[2]
        bars.bar(positions[local] - center, scores[local], color='black', alpha=0.6, width=1)
        bars.axhline(score_threshold, color='gray', linestyle=':', linewidth=1)
        bars.set(xlim=extent[:2], ylim=(0, 1.03), xlabel='Offset from center (nt)',
                 title=f'Score >= {score_threshold:g}: {count} positions')
        bars.set_xticks([-radius, 0, radius] if radius else [0])
        bars.grid(axis='y', alpha=0.2)
        if column == 0:
            bars.set_ylabel('Filtered ensemble score')
        for axis in axes:
            for reference in references:
                if abs(reference - center) <= radius:
                    axis.axvline(reference - center, color='#009E73', linestyle='--', linewidth=1.2)
    figure.tight_layout()
    _save(figure, save_path, dpi)
    return pd.DataFrame(summaries), figure
