"""Regression tests run against either a source or an installed distribution."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from FScanpy import PRFPredictor, extract_prf_regions, fscanr, predict_prf
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
        "predict_sample_examples.csv", "region_example.csv"
    ]
    for name in list_test_data():
        assert Path(get_test_data_path(name)).stat().st_size > 0
    assert len(pd.read_excel(get_test_data_path("full_seq.xlsx"))) == 5


def test_bundled_tutorial_data():
    examples = pd.read_csv(get_test_data_path("predict_sample_examples.csv"))
    assert examples.columns.tolist() == [
        "Sequence_ID", "DNA_seqid", "Genome", "Product", "length",
        "reference_codon_start_0based", "Full_Sequence"
    ]
    assert examples.Sequence_ID.tolist() == list(range(5))
    assert examples.Full_Sequence.str.fullmatch('[ATGC]+').all()
    assert examples.Full_Sequence.str.len().eq(examples.length).all()
    assert examples.reference_codon_start_0based.ge(0).all()
    assert (examples.reference_codon_start_0based + 3 <= examples.length).all()
    assert examples.reference_codon_start_0based.mod(3).eq(0).all()


@pytest.mark.parametrize('sequence_id,count,peak,maximum,reference_count', [
    (0, 260, 300, 0.963053, 5), (1, 235, 645, 0.976088, 6),
    (2, 141, 249, 0.999480, 10), (3, 253, 24, 0.997771, 5),
    (4, 363, 258, 0.969608, None),
])
def test_bundled_tutorial_predictions(predictor, sequence_id, count, peak, maximum, reference_count):
    examples = pd.read_csv(get_test_data_path('predict_sample_examples.csv')).set_index('Sequence_ID')
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
