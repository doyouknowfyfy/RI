import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))      # tests/helpers.py

from helpers import _raw  # noqa: E402
from tsfusion.synthetic import make_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="module")
def synth():
    return make_synthetic_dataset(n_records=400, n_vars=5, seed=1)


@pytest.fixture(scope="module")
def raw_splits(synth):
    return _raw(synth, synth["train_ids"]), _raw(synth, synth["val_ids"])
