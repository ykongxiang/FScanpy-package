"""Tests at the public plotting seams using existing prediction tables."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from FScanpy import plot_prediction_results, plot_prediction_regions, plot_prf_prediction


@pytest.fixture
def table():
    return pd.DataFrame({'Position': [3, 6, 9, 30],
                         'Short_Probability': [0.9, 0.8, 0.1, 0.99],
                         'Long_Probability': [0.9, 0.6, 0.1, 0.99],
                         'Ensemble_Probability': [0.9, 0.72, 0.1, 0.99]})


def test_plotting_an_existing_table_does_not_load_models_or_mutate_it(table, monkeypatch):
    import FScanpy
    def fail(*args, **kwargs):
        raise AssertionError('Plotting results must not load a model')
    monkeypatch.setattr(FScanpy.PRFPredictor, '__init__', fail)
    before = table.copy(deep=True)
    result, figure = plot_prediction_results(table, sequence_length=45, short_threshold=0.2,
                                            long_threshold=0.2, reference_positions=[6],
                                            heatmap_ratios=(0.35,0.35,2.8))
    try:
        assert result is table
        pd.testing.assert_frame_equal(table, before)
        assert len(figure.axes) == 3
        assert [len(a.images) for a in figure.axes] == [1,1,0]
        np.testing.assert_allclose(figure.axes[0].get_subplotspec().get_gridspec().get_height_ratios(), [0.35,0.35,2.8])
        assert all(any(np.array_equal(line.get_xdata(), [6,6]) for line in a.lines) for a in figure.axes)
        assert 'predicted candidates' in figure.axes[0].get_title()
        assert [patch.get_height() for patch in figure.axes[2].patches] == [0.9,0.72,0,0.99]
    finally:
        plt.close(figure)


def test_overlapping_heatmaps_keep_the_maximum_independent_of_row_order():
    table = pd.DataFrame({'Position':[1,2], 'Short_Probability':[0.95,0.3],
                          'Long_Probability':[0.95,0.3], 'Ensemble_Probability':[0.95,0.3]})
    _, a = plot_prediction_results(table, sequence_length=1500, short_threshold=0.2, long_threshold=0.2)
    _, b = plot_prediction_results(table.iloc[::-1], sequence_length=1500, short_threshold=0.2, long_threshold=0.2)
    try:
        assert a.axes[1].images[0].get_array()[0,1] == 0.95
        np.testing.assert_array_equal(a.axes[1].images[0].get_array(), b.axes[1].images[0].get_array())
    finally:
        plt.close(a); plt.close(b)


def test_local_comparisons_use_equal_regions_and_displayed_scores(table):
    summary, figure = plot_prediction_regions(table, [('Reference',6), ('Competitor',30), ('Empty',70)],
                                               radius=3, reference_positions=[6])
    try:
        assert summary.high_score_positions.tolist() == [2,1,0]
        assert summary.scanned_positions.tolist() == [3,1,0]
        np.testing.assert_allclose(summary.local_max.iloc[:2], [0.9,0.99])
        assert np.isnan(summary.local_max.iloc[2])
        assert len(figure.axes) == 9
        assert sum(len(a.images) for a in figure.axes) == 6
        for axis in figure.axes[2::3]:
            np.testing.assert_allclose(axis.get_xlim(), [-3.5,3.5])
            np.testing.assert_allclose(axis.get_ylim(), [0,1.03])
    finally:
        plt.close(figure)


def test_save_path_changes_only_the_file_suffix(table, tmp_path):
    directory = tmp_path / 'folder.png'
    directory.mkdir()
    target = directory / 'plot.PNG'
    _, figure = plot_prediction_results(table, save_path=target, dpi=50)
    try:
        assert target.stat().st_size > 0
        assert target.with_suffix('.pdf').stat().st_size > 0
    finally:
        plt.close(figure)


def test_existing_public_plot_function_accepts_new_keyword_options():
    result, figure = plot_prf_prediction('ATG' * 133, reference_positions=6,
                                         heatmap_ratios=(0.35,0.35,2.8), dpi=50)
    try:
        assert len(result) == 133
        assert figure.axes[0].get_subplotspec().get_gridspec().get_height_ratios() == [0.35,0.35,2.8]
        assert all(len(axis.lines) == 1 for axis in figure.axes)
    finally:
        plt.close(figure)


@pytest.mark.parametrize('options', [{'short_threshold':float('nan')}, {'long_threshold':-1},
                                    {'candidate_threshold':2}, {'heatmap_ratios':(1,0,1)},
                                    {'reference_positions':[-1]}, {'sequence_length':3}])
def test_plot_rejects_invalid_configuration_before_creating_a_figure(table, options):
    count = len(plt.get_fignums())
    with pytest.raises(ValueError):
        plot_prediction_results(table, **options)
    assert len(plt.get_fignums()) == count


def test_multiple_sequences_cannot_silently_share_the_same_position(table):
    with pytest.raises(ValueError, match='one sequence'):
        plot_prediction_results(pd.concat([table, table]))
