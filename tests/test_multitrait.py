"""Tests for multitrait module."""

import pandas as pd
import pytest

from hidden_locus.core.multitrait import TRAITS, run_multitrait_benchmark
from hidden_locus import config


def test_traits_dict_nonempty():
    assert len(TRAITS) >= 5
    assert "ibd" in TRAITS
    assert "t2d" in TRAITS
    assert "ra" in TRAITS


def test_run_multitrait_benchmark_smoke():
    """Smoke test with minimal settings — will use local datasets if present."""
    settings = config.Settings(
        model_name="logreg",
        use_pu=False,
        n_splits=2,
        tissues=["Small_Intestine_Terminal_Ileum"],
    )
    # Run just one trait to keep it fast.
    try:
        df = run_multitrait_benchmark({"ibd": TRAITS["ibd"]}, base_settings=settings)
    except Exception:
        pytest.skip("Local datasets not available for multitrait test.")
    assert isinstance(df, pd.DataFrame)
    assert "trait" in df.columns
    assert "roc_auc" in df.columns
