"""Time Series imputer."""

__maintainer__ = []
__all__ = ["InterpolationImputer", "SimpleImputer"]

from collections.abc import Callable

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import mode

from aeon.transformations.collection.base import BaseCollectionTransformer


class SimpleImputer(BaseCollectionTransformer):
    """Time series imputer.

    Transformer that imputes missing values in time series. Fill values are calculated
    across series.

    Parameters
    ----------
    strategy : str or Callable, default="mean"
        The imputation strategy.
            - if "mean", replace missing values using the mean.
            - if "median", replace missing values using the median.
            - if "constant", replace missing values with the fill_value.
            - if "most frequent", replace missing values with the most frequent value.
            - if Callable, a function that returns the value to replace
            missing values with on each 1D array containing all
            non-missing values of each series.

    fill_value : float or None, default=None
        The value to replace missing values with. Only used when strategy is "constant".
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "removes_missing_values": True,
    }

    def __init__(
        self,
        strategy: str | Callable = "mean",
        fill_value: float | None = None,
    ):
        self.strategy = strategy
        self.fill_value = fill_value
        super().__init__()

    def _transform(
        self, X: np.ndarray | list[np.ndarray], y=None
    ) -> np.ndarray | list[np.ndarray]:
        """
        Transform method to apply the SimpleImputer.

        Parameters
        ----------
        X: np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y: None
            Ignored.

        Returns
        -------
        np.ndarray or list
        """
        self._validate_parameters()

        if isinstance(X, np.ndarray):  # if X is a 3D array
            if self.strategy == "mean":
                X = np.where(np.isnan(X), np.nanmean(X, axis=-1, keepdims=True), X)

            elif self.strategy == "median":
                X = np.where(np.isnan(X), np.nanmedian(X, axis=-1, keepdims=True), X)

            elif self.strategy == "constant":
                X = np.where(np.isnan(X), self.fill_value, X)

            elif self.strategy == "most frequent":
                X = np.where(
                    np.isnan(X),
                    mode(X, axis=-1, nan_policy="omit", keepdims=True).mode,
                    X,
                )

            else:  # if strategy is a callable function
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        nan_mask = np.isnan(X[i, j])
                        X[i, j] = np.where(
                            nan_mask, self.strategy(X[i, j][nan_mask]), X[i, j]
                        )  # applying callable function to each case without nan values
            return X

        else:  # if X is a list of 2D arrays
            Xt = []

            for x in X:
                if self.strategy == "mean":
                    x = np.where(np.isnan(x), np.nanmean(x, axis=-1, keepdims=True), x)
                elif self.strategy == "median":
                    x = np.where(
                        np.isnan(x), np.nanmedian(x, axis=-1, keepdims=True), x
                    )
                elif self.strategy == "constant":
                    x = np.where(np.isnan(x), self.fill_value, x)
                elif self.strategy == "most frequent":
                    x = np.where(
                        np.isnan(x),
                        mode(x, axis=-1, nan_policy="omit", keepdims=True).mode,
                        x,
                    )
                else:  # if strategy is a callable function
                    n_channels = x.shape[0]
                    for i in range(n_channels):
                        nan_mask = np.isnan(x[i])
                        x[i] = np.where(nan_mask, self.strategy(x[i][nan_mask]), x[i])
                Xt.append(x)

            return Xt

    def _validate_parameters(self):
        """Validate the parameters."""
        if self.strategy not in [
            "mean",
            "median",
            "constant",
            "most frequent",
        ] and not callable(self.strategy):
            raise ValueError(
                "strategy must be 'mean', 'median', 'constant', 'most frequent',"
                f" or a callable. Got {self.strategy}."
            )

        if self.strategy == "constant" and self.fill_value is None:
            raise ValueError("fill_value must be provided when strategy is 'constant'.")

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {"strategy": "mean"}


_SPLINE_ORDERS = {"zero": 0, "linear": 1, "slinear": 1, "quadratic": 2, "cubic": 3}
_STEP_STRATEGIES = ("previous", "next", "nearest")


class InterpolationImputer(BaseCollectionTransformer):
    """Time series imputer using interpolation.

    Each channel of each time series is imputed independently, using the
    observed values of that channel and their positions in time. Unlike
    ``SimpleImputer``, which replaces every missing value of a channel with the
    same value, the imputed values follow the local shape of the series.

    Parameters
    ----------
    strategy : str or int, default="linear"
        The interpolation method.
            - "linear" or "slinear": linear interpolation between the closest
              observed values.
            - "quadratic" or "cubic": spline interpolation of order 2 or 3.
            - "zero": spline interpolation of order 0 (piecewise constant).
            - "previous": last observed value (forward fill).
            - "next": next observed value (backward fill).
            - "nearest": closest observed value in time.
            - if int, the order of the spline interpolation.
    extrapolate : bool, default=False
        How the missing values before the first observed value and after the
        last one are imputed with a spline strategy. If False, the first (or
        last) observed value is repeated. If True, the spline is extrapolated,
        which can produce values far outside the range of the series for a
        long gap at the start or the end, in particular for noisy series or
        high-order splines. The "previous", "next" and "nearest" strategies
        always repeat the boundary value.

    See Also
    --------
    SimpleImputer : Impute missing values with a constant value per channel.

    Notes
    -----
    A spline of order ``k`` requires at least ``k + 1`` observed values in each
    channel ("previous", "next", "nearest" and "zero" require one). A
    ``ValueError`` is raised if a channel has fewer observed values. Observed
    values are left unchanged.

    This is a port of ``InterpolationImputer`` from the pyts package [1]_,
    which always extrapolates (``extrapolate=True`` here).

    References
    ----------
    .. [1] J. Faouzi and H. Janati. "pyts: A Python Package for Time Series
       Classification". Journal of Machine Learning Research, 21(46):1-6, 2020.

    Examples
    --------
    >>> import numpy as np
    >>> from aeon.transformations.collection import InterpolationImputer
    >>> X = np.array([[[1.0, np.nan, 3.0, 4.0]], [[8.0, np.nan, 4.0, np.nan]]])
    >>> InterpolationImputer().fit_transform(X)
    array([[[1., 2., 3., 4.]],
    <BLANKLINE>
           [[8., 6., 4., 4.]]])
    >>> InterpolationImputer(extrapolate=True).fit_transform(X)
    array([[[1., 2., 3., 4.]],
    <BLANKLINE>
           [[8., 6., 4., 2.]]])
    """

    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "removes_missing_values": True,
    }

    def __init__(self, strategy: str | int = "linear", extrapolate: bool = False):
        self.strategy = strategy
        self.extrapolate = extrapolate
        super().__init__()

    def _transform(
        self, X: np.ndarray | list[np.ndarray], y=None
    ) -> np.ndarray | list[np.ndarray]:
        """
        Transform method to apply the InterpolationImputer.

        Parameters
        ----------
        X: np.ndarray or list
            Collection to transform. Either a list of 2D arrays with shape
            ``(n_channels, n_timepoints_i)`` or a single 3D array of shape
            ``(n_cases, n_channels, n_timepoints)``.
        y: None
            Ignored.

        Returns
        -------
        np.ndarray or list
            Collection of the same type and shape as ``X``, without missing
            values.
        """
        min_points = self._validate_parameters()
        if isinstance(X, np.ndarray):
            Xt = np.array(X, dtype=np.float64, copy=True)
            for i in range(Xt.shape[0]):
                self._impute_case(Xt[i], i, min_points)
            return Xt
        Xt = []
        for i, x in enumerate(X):
            x = np.array(x, dtype=np.float64, copy=True)
            self._impute_case(x, i, min_points)
            Xt.append(x)
        return Xt

    def _impute_case(self, x: np.ndarray, i: int, min_points: int) -> None:
        """Impute in place the channels of one series of shape (n_channels, n)."""
        positions = np.arange(x.shape[1])
        for j in range(x.shape[0]):
            missing = np.isnan(x[j])
            if not missing.any():
                continue
            n_observed = x.shape[1] - int(missing.sum())
            if n_observed < min_points:
                raise ValueError(
                    f"Channel {j} of case {i} has {n_observed} observed "
                    f"value(s), but strategy={self.strategy!r} requires at "
                    f"least {min_points}."
                )
            observed = ~missing
            x[j, missing] = _interpolate(
                positions[observed],
                x[j, observed],
                positions[missing],
                self.strategy,
                self.extrapolate,
            )

    def _validate_parameters(self) -> int:
        """Validate the parameters and return the minimum number of points."""
        if isinstance(self.strategy, bool) or not (
            isinstance(self.strategy, (int, np.integer))
            or self.strategy in _SPLINE_ORDERS
            or self.strategy in _STEP_STRATEGIES
        ):
            raise ValueError(
                "strategy must be an integer or one of 'linear', 'slinear', "
                "'quadratic', 'cubic', 'zero', 'previous', 'next', 'nearest'. "
                f"Got {self.strategy!r}."
            )
        if isinstance(self.strategy, (int, np.integer)) and self.strategy < 0:
            raise ValueError(
                f"If strategy is an integer, it must be non-negative. Got "
                f"{self.strategy}."
            )
        if not isinstance(self.extrapolate, (bool, np.bool_)):
            raise ValueError(
                f"extrapolate must be a boolean. Got {self.extrapolate!r}."
            )
        if self.strategy in _STEP_STRATEGIES:
            return 1
        return _spline_order(self.strategy) + 1

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return [
            {"strategy": "linear"},
            {"strategy": "cubic", "extrapolate": True},
            {"strategy": "previous"},
        ]


def _spline_order(strategy: str | int) -> int:
    """Order of the spline used by a spline strategy."""
    if isinstance(strategy, (int, np.integer)):
        return int(strategy)
    return _SPLINE_ORDERS[strategy]


def _interpolate(idx, values, new_idx, strategy, extrapolate):
    """Interpolate ``values`` observed at ``idx`` at the positions ``new_idx``."""
    if strategy in _STEP_STRATEGIES:
        return _step_interpolate(idx, values, new_idx, strategy)
    k = _spline_order(strategy)
    if k == 1:
        # Fast path: piecewise linear interpolation, identical to a spline of
        # order 1. np.interp repeats the boundary values outside
        # [idx[0], idx[-1]], so the end segments are extended to extrapolate.
        out = np.interp(new_idx, idx, values)
        if extrapolate and len(idx) >= 2:
            before, after = new_idx < idx[0], new_idx > idx[-1]
            slope_start = (values[1] - values[0]) / (idx[1] - idx[0])
            slope_end = (values[-1] - values[-2]) / (idx[-1] - idx[-2])
            out[before] = values[0] + slope_start * (new_idx[before] - idx[0])
            out[after] = values[-1] + slope_end * (new_idx[after] - idx[-1])
        return out
    spline = make_interp_spline(idx, values, k=k)
    out = spline(new_idx, extrapolate=extrapolate)
    if not extrapolate:
        out[new_idx < idx[0]] = values[0]
        out[new_idx > idx[-1]] = values[-1]
    return out


def _step_interpolate(idx, values, new_idx, strategy):
    """Previous, next or nearest observed value, holding the boundary values."""
    last = len(idx) - 1
    if strategy == "previous":
        pos = np.clip(np.searchsorted(idx, new_idx, side="right") - 1, 0, last)
    elif strategy == "next":
        pos = np.clip(np.searchsorted(idx, new_idx, side="left"), 0, last)
    else:  # "nearest", ties go to the earlier observed value
        right = np.clip(np.searchsorted(idx, new_idx, side="left"), 0, last)
        left = np.clip(right - 1, 0, last)
        pos = np.where(
            np.abs(idx[right] - new_idx) < np.abs(new_idx - idx[left]), right, left
        )
    return values[pos]
