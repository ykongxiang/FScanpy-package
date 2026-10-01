"""Reproduce prediction failures without changing trained model weights."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from FScanpy import PRFPredictor, predict_prf
from FScanpy.features.sequence import SequenceFeatureExtractor
from FScanpy.features.cnn_input import CNNInputProcessor
from FScanpy.utils import extract_window_sequences


@pytest.fixture(scope='module')
def predictor():
    torch.set_num_threads(1)
    return PRFPredictor()


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
