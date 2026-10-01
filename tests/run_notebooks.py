"""Execute the official notebooks in real kernels, outside the source checkout.

Usage: python tests/run_notebooks.py SOURCE OUTPUT
Requires nbclient, nbformat and ipykernel in the current Python environment.
Run in a fresh OUTPUT directory: no adjacent tutorial data are copied.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        parser.error("OUTPUT must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'data').exists(), 'Use a fresh output directory without tutorial data'
    kernel_root = output / "jupyter"
    kernel = kernel_root / "kernels" / "fscanpy-validation"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "FScanpy validation", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(kernel_root)
    os.environ.pop("PYTHONPATH", None)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    results = []
    for filename in ("FScanpy_Demo.ipynb", "tutorial/predict_sample.ipynb",
                     "tutorial/predict_sample_zh.ipynb"):
        nb = nbformat.read(source / filename, as_version=4)
        nbformat.validate(nb)
        # Verify the actual kernel imports the installed wheel, never the checkout.
        probe = nbformat.v4.new_code_cell(
            "import FScanpy, sys\nfrom pathlib import Path\n"
            f"assert Path({str(source)!r}) not in Path(FScanpy.__file__).resolve().parents\n"
            "assert not Path('data').exists() and not Path('tutorial').exists()\n"
            "from FScanpy.data import get_test_data_path\n"
            "assert Path(get_test_data_path('full_seq.xlsx')).is_relative_to(Path(FScanpy.__file__).resolve().parent)\n"
            "print('Installed package:', FScanpy.__file__)\nprint('Kernel:', sys.executable)"
        )
        nb.cells.insert(0, probe)
        if filename == "FScanpy_Demo.ipynb":
            validation = (
                "assert len(fscanr_results) == len(prf_sequences) == len(fscanr_predictions) == 11\n"
                "assert len(validation_predictions) == 3\n"
                "assert len(sequence_results) == 85\n"
            )
        else:
            validation = (
                "import numpy as np\n"
                "expected = {0: (260, 300, 0.963053), 1: (235, 645, 0.976088),\n"
                "            2: (141, 249, 0.999480), 3: (253, 24, 0.997771),\n"
                "            4: (363, 258, 0.969608)}\n"
                "assert set(results_by_sequence) == set(expected)\n"
                "assert 'split' not in examples.columns and 'prf_id' not in examples.columns\n"
                "for sequence_id, (count, position, maximum) in expected.items():\n"
                "    table = results_by_sequence[sequence_id]\n"
                "    assert len(table) == count\n"
                "    assert np.isfinite(table[probability_columns]).all().all()\n"
                "    scores = visible_scores(table)\n"
                "    np.testing.assert_allclose(scores.max(), maximum, atol=5e-4, rtol=0)\n"
                "    assert table.loc[scores.idxmax(), 'Position'] == position\n"
                "    axes = figures_by_sequence[sequence_id].axes\n"
                "    assert len(axes) == 3 and len(axes[0].images) == len(axes[1].images) == 1\n"
                "    assert axes[0].images[0].get_cmap().name == axes[1].images[0].get_cmap().name == 'Reds'\n"
                "    np.testing.assert_allclose(axes[0].get_subplotspec().get_gridspec().get_height_ratios(), [0.35,0.35,2.8])\n"
                "    assert all(any(line.get_color()=='#009E73' for line in axis.lines) for axis in axes)\n"
                "assert len(fine) == 703 and len(fast) == 118\n"
                "np.testing.assert_array_equal(fast.Position, coarse.Position.iloc[::2])\n"
                "np.testing.assert_allclose(fast[probability_columns], coarse[probability_columns].iloc[::2], atol=5e-4, rtol=0)\n"
                "assert len(figure.axes) == 12 and sum(len(axis.images) for axis in figure.axes) == 8\n"
                "assert plot_prediction_regions.__module__ == 'FScanpy.predictor'\n"
            )
        nb.cells.append(nbformat.v4.new_code_cell(validation))
        started = time.monotonic()
        target = output / Path(filename).name
        try:
            NotebookClient(nb, timeout=900, kernel_name="fscanpy-validation",
                           resources={"metadata": {"path": str(output)}}).execute()
        finally:
            nbformat.write(nb, target)
        code_cells = [c for c in nb.cells[1:-1] if c.cell_type == "code"]
        errors = [o for c in code_cells for o in c.outputs if o.output_type == "error"]
        assert not errors
        assert all(c.execution_count is not None for c in code_cells)
        image_count = sum("image/png" in o.get("data", {}) for c in code_cells for o in c.outputs)
        assert image_count > 0, "Notebook produced no inline plots"
        result = {"notebook": filename, "passed_code_cells": len(code_cells),
                  "seconds": round(time.monotonic() - started, 2), "errors": len(errors),
                  "inline_plots": image_count, "prediction_regression": "passed"}
        result['data_source'] = 'installed package'
        result['adjacent_data_directory'] = False
        results.append(result)
        print(json.dumps(result), flush=True)
    (output / "notebook_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
