"""Tests for InterpolationImputer."""

__maintainer__ = []

import numpy as np
import pytest

from aeon.testing.data_generation import (
    make_example_3d_numpy,
    make_example_3d_numpy_list,
)
from aeon.transformations.collection import InterpolationImputer

STRATEGIES = [
    "linear",
    "slinear",
    "quadratic",
    "cubic",
    "zero",
    "previous",
    "next",
    "nearest",
    2,
]


def _with_missing(X, rate=0.2, seed=0):
    """Remove values at random, keeping the first and last 3 of each channel."""
    rng = np.random.RandomState(seed)
    X = X.copy()
    mask = rng.rand(*X.shape) < rate
    mask[..., :3] = False
    mask[..., -3:] = False
    X[mask] = np.nan
    return X, mask


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_3d_numpy(strategy):
    """Missing values are imputed and observed values are unchanged."""
    X, _ = make_example_3d_numpy(
        n_cases=6, n_channels=2, n_timepoints=40, random_state=0
    )
    X_missing, mask = _with_missing(X)
    Xt = InterpolationImputer(strategy=strategy).fit_transform(X_missing)
    assert Xt.shape == X.shape
    assert not np.isnan(Xt).any()
    np.testing.assert_array_equal(Xt[~mask], X[~mask])
    # The input is not modified in place.
    assert np.isnan(X_missing[mask]).all()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_unequal_length_list(strategy):
    """Unequal-length collections give the same result for each series."""
    X, _ = make_example_3d_numpy_list(
        n_cases=5,
        n_channels=2,
        min_n_timepoints=20,
        max_n_timepoints=35,
        random_state=0,
    )
    X_missing = [_with_missing(x, seed=i)[0] for i, x in enumerate(X)]
    imputer = InterpolationImputer(strategy=strategy)
    Xt = imputer.fit_transform(X_missing)
    assert isinstance(Xt, list)
    for x, xt in zip(X_missing, Xt):
        assert xt.shape == x.shape
        assert not np.isnan(xt).any()
        np.testing.assert_array_equal(xt, imputer.fit_transform(x[None])[0])


def test_linear_values():
    """Linear interpolation inside the series and at both ends."""
    X = np.array([[[np.nan, 2.0, np.nan, 6.0, 7.0, np.nan, np.nan]]])
    np.testing.assert_allclose(
        InterpolationImputer().fit_transform(X)[0, 0],
        [0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0],
    )
    np.testing.assert_allclose(
        InterpolationImputer(extrapolate=False).fit_transform(X)[0, 0],
        [2.0, 2.0, 4.0, 6.0, 7.0, 7.0, 7.0],
    )


def test_spline_values():
    """A quadratic spline recovers a quadratic exactly, including at the ends."""
    t = np.arange(10, dtype=float)
    X = (t**2 - 3 * t + 1).reshape(1, 1, -1)
    X_missing = X.copy()
    X_missing[0, 0, [0, 4, 5, 9]] = np.nan
    Xt = InterpolationImputer(strategy="quadratic").fit_transform(X_missing)
    np.testing.assert_allclose(Xt, X)


def test_step_values():
    """Previous, next and nearest observed values."""
    X = np.array([[[np.nan, 1.0, np.nan, np.nan, np.nan, 5.0, np.nan]]])
    expected = {
        "previous": [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0],
        "next": [1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "nearest": [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0],
    }
    for strategy, values in expected.items():
        Xt = InterpolationImputer(strategy=strategy).fit_transform(X)
        np.testing.assert_allclose(Xt[0, 0], values)


@pytest.mark.parametrize("strategy", ["quadratic", "cubic", 4])
def test_no_extrapolation_holds_boundary_values(strategy):
    """With extrapolate=False, the values outside the observed range are held."""
    rng = np.random.RandomState(0)
    X = rng.randn(1, 1, 30)
    X[0, 0, :5] = np.nan
    X[0, 0, -7:] = np.nan
    Xt = InterpolationImputer(strategy=strategy, extrapolate=False).fit_transform(X)
    np.testing.assert_array_equal(Xt[0, 0, :5], X[0, 0, 5])
    np.testing.assert_array_equal(Xt[0, 0, -7:], X[0, 0, -8])


def test_no_missing_values_is_identity():
    """A collection without missing values is returned unchanged."""
    X, _ = make_example_3d_numpy(n_cases=4, n_channels=3, random_state=0)
    np.testing.assert_array_equal(InterpolationImputer().fit_transform(X), X)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "strategy, n_observed",
    [("linear", 1), ("cubic", 3), ("previous", 0), (5, 5)],
)
def test_too_few_observed_values(strategy, n_observed):
    """A channel with too few observed values raises an error."""
    X = np.full((2, 2, 10), np.nan)
    X[0] = 1.0
    X[1, 0] = 1.0
    X[1, 1, :n_observed] = 1.0
    with pytest.raises(ValueError, match="Channel 1 of case 1 has"):
        InterpolationImputer(strategy=strategy).fit_transform(X)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"strategy": "mean"}, "strategy must be"),
        ({"strategy": True}, "strategy must be"),
        ({"strategy": -1}, "non-negative"),
        ({"extrapolate": "yes"}, "extrapolate must be a boolean"),
    ],
)
def test_invalid_parameters(params, match):
    """Invalid parameters raise an error."""
    X = np.array([[[1.0, np.nan, 3.0, 4.0, 5.0]]])
    with pytest.raises(ValueError, match=match):
        InterpolationImputer(**params).fit_transform(X)
