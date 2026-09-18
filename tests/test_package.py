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
        "blastx_example.xlsx", "full_seq.xlsx", "mrna_example.fasta", "region_example.csv"
    ]
    for name in list_test_data():
        assert Path(get_test_data_path(name)).stat().st_size > 0
    assert len(pd.read_excel(get_test_data_path("full_seq.xlsx"))) == 5


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
    assert len(sites) == 16
    extracted = extract_prf_regions(get_test_data_path("mrna_example.fasta"), sites)
    assert len(extracted) == 16
    assert extracted["399bp"].str.len().eq(399).all()
    result = predictor.predict_regions(extracted)
    assert len(result) == 16
    assert np.isfinite(result["Ensemble_Probability"]).all()


def test_plot_accepts_path_object(predictor, tmp_path):
    output = tmp_path / "prediction.png"
    result, figure = predictor.plot_sequence_prediction("ATG" * 12, save_path=output, dpi=50)
    assert len(result) == 12
    assert output.stat().st_size > 0
    assert output.with_suffix(".pdf").stat().st_size > 0
    plt.close(figure)
