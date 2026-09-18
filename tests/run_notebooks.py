"""Execute the official notebooks in real kernels, outside the source checkout.

Usage: python tests/run_notebooks.py SOURCE OUTPUT
Requires nbclient, nbformat and ipykernel in the current Python environment.
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
    for filename in ("FScanpy_Demo.ipynb", "tutorial/predict_sample.ipynb"):
        nb = nbformat.read(source / filename, as_version=4)
        nbformat.validate(nb)
        # Verify the actual kernel imports the installed wheel, never the checkout.
        probe = nbformat.v4.new_code_cell(
            "import FScanpy, sys\nfrom pathlib import Path\n"
            f"assert Path({str(source)!r}) not in Path(FScanpy.__file__).resolve().parents\n"
            "print('Installed package:', FScanpy.__file__)\nprint('Kernel:', sys.executable)"
        )
        nb.cells.insert(0, probe)
        if filename == "FScanpy_Demo.ipynb":
            validation = (
                "assert len(fscanr_results) == len(prf_sequences) == len(fscanr_predictions) == 16\n"
                "assert len(validation_predictions) == 3\n"
                "assert len(sequence_results) == 85\n"
            )
        else:
            validation = (
                "import numpy as np\n"
                "expected = [(sequence_results0, 366, 0.9935215416815536, 9),\n"
                "            (sequence_results1, 5266, 0.9946824525879063, 15),\n"
                "            (sequence_results2, 143, 0.9636074744477399, 105),\n"
                "            (sequence_results3, 143, 0.9823830742659628, 417),\n"
                "            (sequence_results4, 363, 0.9933412495767275, 1053)]\n"
                "for table, count, maximum, position in expected:\n"
                "    assert len(table) == count\n"
                "    assert np.isfinite(table['Ensemble_Probability']).all()\n"
                "    np.testing.assert_allclose(table['Ensemble_Probability'].max(), maximum, atol=1e-6, rtol=1e-6)\n"
                "    assert table.loc[table['Ensemble_Probability'].idxmax(), 'Position'] == position\n"
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
        results.append(result)
        print(json.dumps(result), flush=True)
    (output / "notebook_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
