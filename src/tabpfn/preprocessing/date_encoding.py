#  Copyright (c) Prior Labs GmbH 2026.

"""Resolve every temporal column directly from its dtype, before validation runs.

sklearn's array machinery cannot hold a `datetime64` column beside a numeric
one in one array (no common dtype exists), so a temporal column has to stop
looking like one before `check_array`/`check_X_y` ever run. `DateFeatureExpander`
is where that happens: a point in time (`datetime64`, tz-aware, or `period`) is
either expanded into calendar features via `skrub.DatetimeEncoder` (when
`transform_dates` is on and the column isn't declared categorical) or rendered
to ISO 8601 text (otherwise, so it reads as an ordinary high-cardinality
category downstream). A duration (`timedelta64`) always becomes its length in
seconds -- a quantity with no calendar in it, independent of `transform_dates`.

Because this runs before detection, `detect_feature_modalities` never learns a
column was ever a date at all -- there is no `DATE` modality.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd
from skrub import DatetimeEncoder

from tabpfn.preprocessing.datamodel import make_names_unique
from tabpfn.preprocessing.modality_detection import _format_names_for_warning

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from tabpfn.constants import XType


def _is_datetime_like_dtype(dtype: Any) -> bool:
    """Whether `dtype` holds points in time: `datetime64`, tz-aware, or `period`."""
    return pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(
        dtype, pd.PeriodDtype
    )


def _as_timestamp(column: pd.Series) -> pd.Series:
    """The instant a `PeriodDtype` column starts at, or the column unchanged.

    A period is a span, not an instant; its start is the instant that orders
    identically, which is all a calendar feature or a rendered string needs.
    """
    if isinstance(column.dtype, pd.PeriodDtype):
        return column.dt.to_timestamp()
    return column


def _make_datetime_encoder() -> DatetimeEncoder:
    """Build the encoder that turns a datetime column into calendar features.

    Returns:
        An encoder producing the year, the day of year, the seconds since epoch,
        and the cyclical month, day and weekday pairs, plus the time of day when
        the column carries one.
    """
    return DatetimeEncoder(
        resolution="second",
        add_weekday=True,
        add_day_of_year=True,
        periodic_encoding="circular",
    )


def _replace_columns_positionally(
    X: pd.DataFrame,
    replacements: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Return `X` with the given column positions replaced, leaving `X` untouched.

    Positional, and via a temporary integer column axis rather than
    ``isetitem``: the labels are the caller's, so they can repeat, which makes
    assignment by label ambiguous -- and ``isetitem`` only arrived in pandas
    1.5, below this package's floor. Numbering the axis makes every label unique
    and equal to its own position, so a plain assignment is unambiguous, and the
    caller's labels go back afterwards.

    The copy is shallow and the frame handed in is never written through: each
    assignment replaces a whole column rather than any value inside one.
    """
    out = X.copy(deep=False)
    original_columns = out.columns
    out.columns = pd.RangeIndex(out.shape[1])
    for position, values in replacements.items():
        out[position] = values
    out.columns = original_columns
    return out


def _warn_on_dates(column_names: Sequence[str]) -> None:
    """Warn about date columns that will be read as a plain category or text.

    Empty whenever every unexpanded date was declared categorical.
    """
    if not column_names:
        return
    warnings.warn(
        f"These columns hold dates, which are read as plain categories or "
        f"text: {_format_names_for_warning(list(column_names))}.\n"
        'Raise `inference_config={"TRANSFORM_DATES": True}` to expand them into '
        "calendar features instead. To silence this for a column that should "
        "stay a plain category or text, pass its index in "
        "`categorical_features_indices`.",
        UserWarning,
        # stacklevel=6 reaches the `estimator.fit(X, y)` call site; pinned by the
        # `warning.filename` assert in the tests.
        stacklevel=6,
    )


class DateFeatureExpander:
    """Resolves every temporal column: expands it, or renders it to text.

    Not a `PreprocessingStep` (`pipeline_interface.py`): that tier runs per
    ensemble member on already-numeric arrays, well past where this needs to
    run. Not `BaseEstimator`/`TransformerMixin` either: `fit_transform` takes
    extra fitting parameters that don't fit sklearn's `fit(X, y=None)` shape,
    and returns more than just the transformed data.

    Usage mirrors `ordinal_encoder_`: construct one, call `fit_transform` once
    at fit time and keep the instance around (e.g. as `self.date_expander_`),
    then call `transform` on it at predict time.
    """

    @dataclasses.dataclass
    class _FittedColumn:
        """A fitted `DatetimeEncoder` for one column, and its output names."""

        encoder: DatetimeEncoder
        output_names: list[str]

    def __init__(self) -> None:
        self._fitted: dict[int, DateFeatureExpander._FittedColumn] = {}

    @property
    def expanded_indices(self) -> list[int]:
        """Raw column indices that were expanded, ascending.

        Empty both before `fit_transform` is called and after it finds no
        column eligible to expand.
        """
        return sorted(self._fitted)

    def fit_transform(
        self,
        X: XType,
        *,
        transform_dates: bool = False,
        categorical_features_indices: Sequence[int] = (),
    ) -> tuple[XType, list[str] | None, list[int], list[int], list[int]]:
        """Resolve every temporal column in `X`, before any validation runs.

        A point in time (`datetime64`, tz-aware, or `period`) is expanded into
        calendar features when `transform_dates` is on and it isn't declared
        categorical; otherwise it is rendered to ISO 8601 text. A duration
        (`timedelta64`) always becomes its length in seconds. Not a
        `DataFrame`, or holding neither: a no-op.

        Args:
            X: The input data, before any dtype fixing.
            transform_dates: Whether an eligible date column is expanded
                rather than rendered to text.
            categorical_features_indices: Indices the caller declared
                categorical; a date column among them is never expanded,
                regardless of `transform_dates`.

        Returns:
            A 5-tuple: the resolved data; the raw (unprefixed) names for its
            columns, in order (`None` if `X` wasn't a `DataFrame`);
            `categorical_features_indices` remapped onto the resolved
            columns; the resolved columns' positions that hold calendar-
            expansion output (always numerical, never subject to the
            cardinality heuristic downstream); and the resolved columns'
            positions holding a date rendered to text (already warned about
            by name here, so the caller's own free-text warning must not
            report them again).
        """
        self._fitted = {}
        categorical_features_indices = list(categorical_features_indices)
        if not isinstance(X, pd.DataFrame):
            return X, None, categorical_features_indices, [], []

        dtypes = list(X.dtypes)
        date_indices = [
            i for i, dtype in enumerate(dtypes) if _is_datetime_like_dtype(dtype)
        ]
        duration_indices = [
            i
            for i, dtype in enumerate(dtypes)
            if pd.api.types.is_timedelta64_dtype(dtype)
        ]
        if not date_indices and not duration_indices:
            return (
                X,
                [str(c) for c in X.columns],
                categorical_features_indices,
                [],
                [],
            )

        categorical = set(categorical_features_indices)
        to_expand = [
            i for i in date_indices if transform_dates and i not in categorical
        ]
        rendered_as_text = [
            i for i in date_indices if i not in to_expand and i not in categorical
        ]
        _warn_on_dates([str(X.columns[i]) for i in rendered_as_text])

        single_column_replacements: dict[int, np.ndarray] = {}
        for position in date_indices:
            if position in to_expand:
                continue
            column = _as_timestamp(X.iloc[:, position])
            single_column_replacements[position] = (
                column.astype(str).where(column.notna(), None).to_numpy()
            )
        for position in duration_indices:
            single_column_replacements[position] = (
                X.iloc[:, position].dt.total_seconds().to_numpy()
            )
        X = _replace_columns_positionally(X, single_column_replacements)

        def _remap(indices: Sequence[int]) -> list[int]:
            return [i - sum(1 for j in to_expand if j < i) for i in indices]

        if not to_expand:
            return (
                X,
                [str(c) for c in X.columns],
                _remap(categorical_features_indices),
                [],
                _remap(rendered_as_text),
            )

        # Only reached when there is something to expand: `_assemble` below
        # concatenates `X`'s kept columns against skrub's own (freshly
        # default-indexed) output, so `X`'s row index must already be the
        # default range or the two would misalign by label instead of position.
        X = X.reset_index(drop=True)
        existing_names = [str(c) for c in X.columns]
        encoded_blocks: list[pd.DataFrame] = []
        for position in to_expand:
            name = str(X.columns[position])
            column = _as_timestamp(X.iloc[:, position]).rename(name)
            encoded, fitted_column = self._fit_one_column(column, existing_names)
            existing_names += fitted_column.output_names
            self._fitted[position] = fitted_column
            encoded_blocks.append(encoded.reset_index(drop=True))

        out = self._assemble(X, to_expand, encoded_blocks)

        expand_set = set(to_expand)
        kept_names = [str(c) for i, c in enumerate(X.columns) if i not in expand_set]
        expanded_names = [
            name
            for position in to_expand
            for name in self._fitted[position].output_names
        ]
        numerical_hint_indices = list(
            range(len(kept_names), len(kept_names) + len(expanded_names))
        )
        return (
            out,
            kept_names + expanded_names,
            _remap(categorical_features_indices),
            numerical_hint_indices,
            _remap(rendered_as_text),
        )

    def transform(self, X: XType) -> XType:
        """Reapply the resolution decided by `fit_transform`, positionally.

        Only `expanded_indices` (frozen at fit time) are ever expanded -- never
        re-decided here. A position that was expanded at fit time but is no
        longer a genuine datetime dtype right now degrades to a `NaN` calendar
        feature, the same as any other missing value: there is no attempt to
        parse it from whatever is actually sitting there instead.

        Args:
            X: The data, before any dtype fixing.
        """
        if not isinstance(X, pd.DataFrame):
            return X
        to_expand = self.expanded_indices
        dtypes = list(X.dtypes)
        date_indices = [
            i for i, dtype in enumerate(dtypes) if _is_datetime_like_dtype(dtype)
        ]
        duration_indices = [
            i
            for i, dtype in enumerate(dtypes)
            if pd.api.types.is_timedelta64_dtype(dtype)
        ]
        if not to_expand and not date_indices and not duration_indices:
            return X

        expand_set = set(to_expand)
        single_column_replacements: dict[int, np.ndarray] = {}
        for position in date_indices:
            if position in expand_set:
                continue
            column = _as_timestamp(X.iloc[:, position])
            single_column_replacements[position] = (
                column.astype(str).where(column.notna(), None).to_numpy()
            )
        for position in duration_indices:
            single_column_replacements[position] = (
                X.iloc[:, position].dt.total_seconds().to_numpy()
            )
        X = _replace_columns_positionally(X, single_column_replacements)

        if not to_expand:
            return X

        # Only reached when there is something to expand -- see the identical
        # comment in `fit_transform`.
        X = X.reset_index(drop=True)
        encoded_blocks = []
        for position in to_expand:
            fitted_column = self._fitted[position]
            if _is_datetime_like_dtype(dtypes[position]):
                column = _as_timestamp(X.iloc[:, position])
            else:
                column = pd.Series(pd.NaT, index=range(len(X)), dtype="datetime64[ns]")
            encoded = self._apply_one_column(column, fitted_column)
            encoded_blocks.append(encoded.reset_index(drop=True))

        return self._assemble(X, to_expand, encoded_blocks)

    @staticmethod
    def _assemble(
        frame: pd.DataFrame,
        to_expand: list[int],
        encoded_blocks: list[pd.DataFrame],
    ) -> pd.DataFrame:
        # Positional, not `frame.drop(columns=...)`: column labels can repeat
        # (pandas allows duplicate names), so dropping by label instead of
        # position would misbehave in that case.
        keep = [i for i in range(frame.shape[1]) if i not in set(to_expand)]
        remaining = frame.iloc[:, keep]
        return pd.concat([remaining, *encoded_blocks], axis=1)

    @staticmethod
    def _fit_one_column(
        column: pd.Series,
        existing_names: list[str],
    ) -> tuple[pd.DataFrame, DateFeatureExpander._FittedColumn]:
        """Fit a new encoder on one column.

        `column` must already carry the real feature name: skrub names each
        output after it (e.g. "signed_on_year", "signed_on_month_circular_0"),
        and those are kept as-is here rather than replaced with a generic
        "{name}_{i}", deduped only against name collisions with existing
        columns.
        """
        encoder = _make_datetime_encoder()
        raw_encoded = pd.DataFrame(encoder.fit_transform(column))
        output_names = make_names_unique(
            list(raw_encoded.columns), existing=existing_names
        )
        encoded = raw_encoded.set_axis(output_names, axis=1)
        fitted_column = DateFeatureExpander._FittedColumn(
            encoder=encoder, output_names=output_names
        )
        return encoded, fitted_column

    @staticmethod
    def _apply_one_column(
        column: pd.Series,
        fitted: DateFeatureExpander._FittedColumn,
    ) -> pd.DataFrame:
        """Reapply an already-fitted encoder to one column."""
        encoded = fitted.encoder.transform(column)
        return pd.DataFrame(encoded).set_axis(fitted.output_names, axis=1)


def apply_date_expansion(X: XType, source: object) -> XType:
    """Resolve `X`'s temporal columns via `source`'s fitted `date_expander_`.

    `source` (a fitted estimator or ensemble worker) may never have set
    `date_expander_` at all -- e.g. `fit_from_preprocessed` skips the step
    that would. A fresh, nothing-fitted expander still renders any date
    column to text and any duration column to seconds, covering that case
    with no special handling.
    """
    date_expander = getattr(source, "date_expander_", None) or DateFeatureExpander()
    return date_expander.transform(X)
