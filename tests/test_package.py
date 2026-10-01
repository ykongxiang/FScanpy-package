"""Regression tests run against either a source or an installed distribution."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from FScanpy import PRFPredictor, extract_prf_regions, fscanr, predict_prf, plot_prediction_results, plot_prediction_regions, plot_prf_prediction
from FScanpy.features.sequence import SequenceFeatureExtractor
from FScanpy.features.cnn_input import CNNInputProcessor
from FScanpy.utils import extract_window_sequences
from Bio.Seq import Seq
from FScanpy.data import get_test_data_path, list_test_data


@pytest.fixture(scope="module")
def predictor():
    torch.set_num_threads(1)
    return PRFPredictor()


@pytest.fixture(scope="module")
def regions():
    return pd.read_csv(get_test_data_path("region_example.csv"))


def test_bundled_data():
    assert list_test_data() == [
        "blastx_example.xlsx", "full_seq.xlsx", "mrna_example.fasta",
        "region_example.csv"
    ]
    for name in list_test_data():
        assert Path(get_test_data_path(name)).stat().st_size > 0
    assert len(pd.read_excel(get_test_data_path("full_seq.xlsx"))) == 5


def test_bundled_tutorial_data():
    examples = pd.read_excel(get_test_data_path("full_seq.xlsx")).rename(columns={"Position":"reference_codon_start_0based"})
    assert set(['Sequence_ID', 'Full_Sequence', '33bp', '399bp', 'length', 'reference_codon_start_0based']).issubset(examples.columns)
    assert examples.Sequence_ID.tolist() == list(range(5))
    assert examples.Full_Sequence.str.fullmatch('[ATGC]+').all()
    assert examples.Full_Sequence.str.len().eq(examples.length).all()
    assert examples.reference_codon_start_0based.ge(0).all()
    assert (examples.reference_codon_start_0based + 3 <= examples.length).all()
    assert examples.reference_codon_start_0based.mod(3).eq(0).all()
    for _, row in examples.iterrows():
        assert extract_window_sequences(row.Full_Sequence, row.reference_codon_start_0based) == (row['33bp'], row['399bp'])


@pytest.mark.parametrize('sequence_id,count,peak,maximum,reference_count', [
    (0, 260, 300, 0.963053, 5), (1, 235, 645, 0.976088, 6),
    (2, 141, 249, 0.999480, 10), (3, 253, 24, 0.997771, 5),
    (4, 363, 258, 0.969608, None),
])
def test_bundled_tutorial_predictions(predictor, sequence_id, count, peak, maximum, reference_count):
    examples = pd.read_excel(get_test_data_path('full_seq.xlsx')).rename(columns={'Position':'reference_codon_start_0based'}).set_index('Sequence_ID')
    row = examples.loc[sequence_id]
    result = predictor.predict_sequence(row.Full_Sequence, window_size=3,
                                        short_threshold=0.1, ensemble_weight=0.6)
    scores = result.Ensemble_Probability.where(
        (result.Short_Probability >= 0.2) & (result.Long_Probability >= 0.2), 0)
    assert len(result) == count
    assert result.loc[scores.idxmax(), 'Position'] == peak
    np.testing.assert_allclose(scores.max(), maximum, atol=5e-4, rtol=0)
    if reference_count is not None:
        local = (result.Position - row.reference_codon_start_0based).abs() <= 15
        assert int((scores[local] >= 0.7).sum()) == reference_count


@pytest.mark.parametrize("column", ["Long_Sequence", "399bp"])
def test_dataframe_regions_match_series(predictor, regions, column):
    series = regions["399bp"]
    expected = predictor.predict_regions(series)
    actual = predictor.predict_regions(pd.DataFrame({column: series}))
    pd.testing.assert_frame_equal(actual, expected)


def test_dataframe_missing_sequence_column(predictor):
    with pytest.raises(Exception, match="Long_Sequence.*399bp"):
        predictor.predict_regions(pd.DataFrame({"other": ["ATG"]}))


def test_official_region_predictions_unchanged(predictor, regions, capsys):
    # Recorded from upstream d4a8696669a68, CPU, scikit-learn 1.7.2.
    expected = np.array([
        [0.0041030961619417305, 0.0, 0.0],
        [0.430870618553921, 0.11595644056797028, 0.24192211176235057],
        [0.7608684099830634, 0.734519898891449, 0.7450593033280948],
    ])
    result = predictor.predict_regions(regions["399bp"])
    columns = ["Short_Probability", "Long_Probability", "Ensemble_Probability"]
    np.testing.assert_allclose(result[columns], expected, atol=1e-6, rtol=1e-6)
    assert capsys.readouterr().out == ""


def test_public_dataframe_preserves_metadata(regions):
    result = predict_prf(data=regions)
    assert len(result) == len(regions)
    pd.testing.assert_series_equal(result["DNA_seqid"], regions["DNA_seqid"])


def test_blastx_pipeline_and_public_exports(predictor):
    blastx = pd.read_excel(get_test_data_path("blastx_example.xlsx"))
    sites = fscanr(blastx, mismatch_cutoff=10, evalue_cutoff=1e-5, frameDist_cutoff=10)
    # Correct peptide-coordinate deduplication matches the original R algorithm.
    assert len(sites) == 11
    extracted = extract_prf_regions(get_test_data_path("mrna_example.fasta"), sites)
    assert len(extracted) == 11
    assert extracted["399bp"].str.len().eq(399).all()
    result = predictor.predict_regions(extracted)
    assert len(result) == 11
    assert np.isfinite(result["Ensemble_Probability"]).all()


def test_plot_accepts_path_object(predictor, tmp_path):
    output = tmp_path / "prediction.png"
    result, figure = predictor.plot_sequence_prediction("ATG" * 12, save_path=output, dpi=50)
    assert len(result) == 12
    assert output.stat().st_size > 0
    assert output.with_suffix(".pdf").stat().st_size > 0
    plt.close(figure)


def test_tiny_sequence_windows_preserve_scanned_codon():
    short, long = extract_window_sequences('ATG', 0)
    assert short == 'N' * 16 + 'ATG' + 'N' * 14
    assert long == 'N' * 199 + 'ATG' + 'N' * 197



def test_even_length_input_is_trimmed_to_exact_short_length():
    extractor = SequenceFeatureExtractor()
    source = 'ATGC' * 9
    assert extractor.trim_sequence(source[:34], 33) == source[:33]



def test_cnn_input_one_extra_base_is_encoded_instead_of_zeroed():
    processor = CNNInputProcessor()
    source = 'T' * 400
    encoded = processor.prepare_sequence(source)
    assert encoded.shape == (1, 399, 1)
    assert np.all(encoded == 1)



def test_short_feature_vector_has_the_trained_dimension():
    extractor = SequenceFeatureExtractor()
    assert len(extractor.extract_features('ATG')) == len(extractor.feature_names)



def test_model_failure_is_not_reported_as_zero_probability(predictor, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError('broken classifier')
    monkeypatch.setattr(predictor.short_model, 'predict_proba', failure)
    with pytest.raises(RuntimeError, match='[Ss]hort.*broken classifier'):
        predictor.predict_single_position('ATG' * 11, 'ATG' * 133)



def test_low_plot_display_threshold_does_not_skip_long_prediction(predictor, monkeypatch):
    monkeypatch.setattr(predictor, '_predict_model', lambda *args: 0.05)
    monkeypatch.setattr(predictor, '_predict_long_torch', lambda seq: 0.8)
    result, figure = predictor.plot_sequence_prediction(
        'ATG' * 3, short_threshold=0.02, long_threshold=0.02)
    try:
        np.testing.assert_allclose(result.Long_Probability, 0.8)
        assert max(patch.get_height() for patch in figure.axes[2].patches) > 0
    finally:
        plt.close(figure)



def test_metadata_does_not_overwrite_computed_probabilities():
    data = pd.DataFrame({'399bp': ['ATG' * 133], 'Ensemble_Probability': [-1.0], 'label': ['example']})
    expected = predict_prf(data=data.drop(columns='Ensemble_Probability'))
    actual = predict_prf(data=data)
    np.testing.assert_allclose(actual.Ensemble_Probability, expected.Ensemble_Probability)
    assert actual.label.tolist() == ['example']



def test_rna_uracil_is_normalized_consistently_for_long_model():
    assert PRFPredictor._process_sequence('augc') == 'ATGC'



def test_region_model_failure_reaches_the_public_caller(predictor, monkeypatch):
    import FScanpy
    monkeypatch.setattr(FScanpy, 'PRFPredictor', lambda **kwargs: predictor)
    monkeypatch.setattr(predictor, '_predict_model', lambda *args: 0.8)
    def failure(*args):
        raise RuntimeError('broken long weights')
    monkeypatch.setattr(predictor, '_predict_long_torch', failure)
    with pytest.raises(RuntimeError, match='region 1.*broken long weights'):
        predict_prf(data=pd.DataFrame({'399bp': ['ATG' * 133]}))



def test_per_position_scan_reuses_windows_without_dropping_rows(predictor, monkeypatch):
    calls = []
    def predict(short, long, *args):
        calls.append((short, long))
        return {'Short_Probability': 0.8, 'Long_Probability': 0.9,
                'Ensemble_Probability': 0.86, 'Ensemble_Weights': 'Short:0.4, Long:0.6'}
    monkeypatch.setattr(predictor, 'predict_single_position', predict)
    table = predictor.predict_sequence('ATG' * 3, window_size=1)
    assert len(calls) == 3
    assert table.Position.tolist() == list(range(7))
    assert table.Codon.tolist() == ['ATG', 'TGA', 'GAT', 'ATG', 'TGA', 'GAT', 'ATG']
    assert table.groupby(table.Position // 3).Long_Sequence.nunique().eq(1).all()



@pytest.mark.parametrize('window_size', [0, -1, 1.5, True, float('nan')])
def test_scan_interval_is_a_positive_integer(predictor, window_size):
    with pytest.raises(ValueError, match='window_size'):
        predictor.predict_sequence('ATG' * 3, window_size=window_size)



@pytest.mark.parametrize('length', [3, 30, 120, 399, 705])
def test_short_and_long_windows_keep_the_requested_codon_at_their_center(length):
    sequence = 'ATG' * (length // 3)
    for position in [0, len(sequence)//6*3, len(sequence)-3]:
        short, long = extract_window_sequences(sequence, position)
        assert short[16:19] == long[199:202] == 'ATG'
        assert len(short) == 33 and len(long) == 399



def test_documented_feature_and_model_info_methods_exist(predictor):
    features = predictor.extract_features(['ATG' * 11, 'ATG'])
    assert features.shape == (2, len(predictor.feature_extractor.feature_names))
    info = predictor.get_model_info()
    assert info['short_model'] == 'HistGradientBoostingClassifier'
    assert info['backend'] == 'pytorch'
    assert info['short_input_bp'] == 33 and info['long_input_bp'] == 399



def test_ambiguous_bases_use_the_same_N_convention_in_both_models():
    extractor = SequenceFeatureExtractor()
    source = 'ATGR' * 8 + 'A'
    np.testing.assert_array_equal(extractor.extract_features(source),
                                  extractor.extract_features(PRFPredictor._process_sequence(source)))



def test_legacy_feature_prediction_helper_does_not_silently_return_no_rows():
    extractor = SequenceFeatureExtractor()
    data = pd.DataFrame({'33bp':['ATG' * 11], '399bp':['ATG' * 133]})
    with pytest.warns(DeprecationWarning, match='predict_regions'):
        result = extractor.predict_region_batch(data)
    assert len(result) == 1
    np.testing.assert_allclose(result.Ensemble_Probability, predict_prf(data=data).Ensemble_Probability)



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



def test_different_peptide_sites_are_not_deduplicated_by_dna_coordinates():
    rows = []
    for name, peptide_start in [('gene_a', 1), ('gene_b', 18)]:
        rows.extend([
            [name,'protein',99,33,0,0,1,99,peptide_start,peptide_start+32,1e-20,100,1,0],
            [name,'protein',99,30,0,0,101,190,peptide_start+33,peptide_start+62,1e-20,100,2,0],
        ])
    result = fscanr(pd.DataFrame(rows))
    assert result.DNA_seqid.tolist() == ['gene_a','gene_b']
    assert result.Pep_FS_start.tolist() == [34,51]



def test_peptide_gap_cutoff_uses_amino_acids_not_nucleotides():
    rows = [
        ['gene','protein',99,33,0,0,1,99,1,33,1e-20,100,1,0],
        ['gene','protein',99,30,0,0,110,199,37,66,1e-20,100,2,0],
    ]
    assert fscanr(pd.DataFrame(rows),frameDist_cutoff=10).empty



@pytest.mark.parametrize('strand', ['+','-'])
def test_explicit_zero_based_input_uses_the_same_physical_position(tmp_path, strand):
    sequence = 'ACGT' * 225
    fasta = tmp_path / 'sequence.fasta'
    fasta.write_text('>gene\n'+sequence+'\n')
    sites = pd.DataFrame({'DNA_seqid':['gene'], 'FS_start':[603], 'FS_end':[604],
                          'Strand':[strand], 'FS_type':[1]})
    expected = extract_prf_regions(str(fasta),sites)
    zero_based = sites.assign(FS_start=sites.FS_start-1, FS_end=sites.FS_end-1)
    actual = extract_prf_regions(str(fasta),zero_based,coordinate_base=0)
    assert expected['399bp'].tolist() == actual['399bp'].tolist()



@pytest.mark.parametrize('strand, position', [('+',603), ('-',734)])
def test_blastx_position_is_converted_to_coding_strand_before_window_extraction(tmp_path, strand, position):
    sequence = 'ACGT' * 225
    fasta = tmp_path / 'sequence.fasta'
    fasta.write_text('>gene\n'+sequence+'\n')
    sites = pd.DataFrame({'DNA_seqid':['gene'], 'FS_start':[position], 'FS_end':[position+1],
                          'Strand':[strand], 'FS_type':[1]})
    result = extract_prf_regions(str(fasta),sites)
    oriented = sequence if strand=='+' else str(Seq(sequence).reverse_complement())
    converted = position-1 if strand=='+' else len(sequence)-position
    assert result['399bp'].iloc[0] == extract_window_sequences(oriented,converted)[1]
    assert result.FS_start.iloc[0] == position
