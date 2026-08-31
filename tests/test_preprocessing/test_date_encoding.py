#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `DateFeatureExpander`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import tabpfn.base
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.base import get_embeddings
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.date_encoding import DateFeatureExpander, _warn_on_dates

N = 20


def _numeric_and_date_frame(n: int = N) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": np.arange(n, dtype=float),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )


def test__not_a_dataframe__is_a_noop() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    expander = DateFeatureExpander()
    X_out, names, cat, num_hint, text_hint = expander.fit_transform(
        X, categorical_features_indices=[0]
    )
    assert X_out is X
    assert names is None
    assert cat == [0]
    assert num_hint == []
    assert text_hint == []
    assert expander.expanded_indices == []


def test__no_temporal_columns__is_a_noop() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    X_out, names, cat, num_hint, text_hint = DateFeatureExpander().fit_transform(X)
    assert X_out is X
    assert names == ["a", "b"]
    assert cat == []
    assert num_hint == []
    assert text_hint == []


class TestRenderToText:
    """`transform_dates=False` (the default): a date column becomes text."""

    def test__date_not_transformed__is_rendered_to_text_and_reported(self) -> None:
        X = _numeric_and_date_frame()
        with pytest.warns(UserWarning, match="hold dates"):
            X_out, names, _, num_hint, text_hint = DateFeatureExpander().fit_transform(
                X, transform_dates=False
            )
        assert text_hint == [1]
        assert num_hint == []
        assert names == ["num", "signed_on"]
        assert X_out.iloc[0, 1] == "2020-01-01"
        assert X_out.shape == X.shape

    @pytest.mark.parametrize(
        ("label", "column", "expected_first"),
        [
            ("datetime64", pd.date_range("2020-01-01", periods=3), "2020-01-01"),
            (
                "tz aware",
                pd.date_range("2020-01-01", periods=3, tz="UTC"),
                "2020-01-01 00:00:00+00:00",
            ),
            (
                "with time",
                pd.date_range("2020-01-01 13:45", periods=3, freq="D"),
                "2020-01-01 13:45:00",
            ),
            (
                "period",
                pd.date_range("2020-01-01", periods=3).to_period("M"),
                "2020-01-01",
            ),
        ],
    )
    def test__date_columns__render_as_text(
        self, label: str, column: pd.Index, expected_first: str
    ) -> None:
        X = pd.DataFrame({"n": [1.0, 2.0, 3.0], "d": column})
        with pytest.warns(UserWarning, match="hold dates"):
            out, *_ = DateFeatureExpander().fit_transform(X)
        assert out.iloc[0, 1] == expected_first, label

    def test__missing_stays_missing__not_the_string_nat(self) -> None:
        """`astype(str)` alone writes `NaT` out as the literal `"NaT"`."""
        column = pd.Series(pd.to_datetime(["2020-01-01", None, "2020-01-03"]))
        with pytest.warns(UserWarning, match="hold dates"):
            out, *_ = DateFeatureExpander().fit_transform(pd.DataFrame({"d": column}))
        assert out["d"].isna().tolist() == [False, True, False]
        assert not (out["d"] == "NaT").any()

    def test__timedelta__becomes_seconds_and_is_not_a_date(self) -> None:
        """A duration is a quantity, not a point on a calendar."""
        X = pd.DataFrame({"d": pd.to_timedelta([1, 2, 3], unit="D")})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out, _, _, _, text_hint = DateFeatureExpander().fit_transform(X)
        assert text_hint == []
        assert out["d"].tolist() == [86400.0, 172800.0, 259200.0]

    def test__input_frame_is_not_mutated(self) -> None:
        X = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=3)})
        before = X.copy(deep=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            DateFeatureExpander().fit_transform(X)
        pd.testing.assert_frame_equal(X, before)

    def test__duplicate_column_labels__are_replaced_by_position(self) -> None:
        """Labels can repeat (pandas allows it), so replacement must be
        positional, never by label.
        """
        X = pd.concat(
            [
                pd.Series([1.0, 2.0, 3.0]),
                pd.Series(pd.date_range("2020-01-01", periods=3)),
                pd.Series([4.0, 5.0, 6.0]),
            ],
            axis=1,
        )
        X.columns = ["same", "same", "same"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, *_ = DateFeatureExpander().fit_transform(X)
        assert out.iloc[0, 1] == "2020-01-01"
        assert out.iloc[:, 0].tolist() == [1.0, 2.0, 3.0]
        assert out.iloc[:, 2].tolist() == [4.0, 5.0, 6.0]

    def test__non_unique_index__is_preserved_when_not_expanding(self) -> None:
        X = pd.DataFrame(
            {"n": [1.0, 2.0, 3.0], "d": pd.date_range("2020-01-01", periods=3)},
            index=[7, 7, 2],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, *_ = DateFeatureExpander().fit_transform(X)
        assert list(out.index) == [7, 7, 2]
        assert out.iloc[0, 1] == "2020-01-01"

    def test__values_of_the_input_frame_are_not_written_through(self) -> None:
        X = pd.DataFrame({"n": [1.0, 2.0], "d": pd.date_range("2020-01-01", periods=2)})
        before = X.copy(deep=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            DateFeatureExpander().fit_transform(X)
        pd.testing.assert_frame_equal(X, before)

    def test__rendering_twice__is_a_noop_the_second_time(self) -> None:
        """Predict re-renders whatever it is handed, including a frame fit
        already rendered, so the second pass must find nothing left to do.
        """
        X = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=3)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            once, *_ = DateFeatureExpander().fit_transform(X)
            twice, *_ = DateFeatureExpander().fit_transform(once)
        assert twice is once


class TestExpand:
    """`transform_dates=True`: an eligible date column expands into numbers."""

    def test__fit__removes_raw_column_and_appends_numeric_features(self) -> None:
        X = _numeric_and_date_frame()
        expander = DateFeatureExpander()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            X_out, names, _, num_hint, text_hint = expander.fit_transform(
                X, transform_dates=True
            )

        assert text_hint == []
        assert expander.expanded_indices == [1]
        assert X_out.shape[0] == N
        assert X_out.shape[1] > 2
        assert num_hint == list(range(1, X_out.shape[1]))
        assert names[0] == "num"
        assert all(name.startswith("signed_on_") for name in names[1:])
        # Every expanded feature is real-valued for a fully populated date column.
        assert np.isfinite(X_out.iloc[:, 1:].to_numpy(dtype=float)).all()

    def test__output_names_are_skrubs_own_descriptive_names(self) -> None:
        """Skrub's own per-feature names (e.g. "_year", "_month_circular_0")
        are kept as-is, not replaced with a generic "_0", "_1", ...
        """
        X = _numeric_and_date_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, names, *_ = DateFeatureExpander().fit_transform(X, transform_dates=True)

        assert names[1:] == [
            "signed_on_year",
            "signed_on_total_seconds",
            "signed_on_day_of_year",
            "signed_on_month_circular_0",
            "signed_on_month_circular_1",
            "signed_on_day_circular_0",
            "signed_on_day_circular_1",
            "signed_on_weekday_circular_0",
            "signed_on_weekday_circular_1",
        ]

    def test__every_date_column__is_expanded_unconditionally(self) -> None:
        """Nothing is weighed here: a column eligible to expand does, full
        stop. Which columns are eligible -- including a declared categorical,
        which is never eligible -- is decided just above, in this same call.
        """
        n = N
        X = pd.DataFrame(
            {
                "a": pd.date_range("2020-01-01", periods=n, freq="D"),
                "b": pd.date_range("2021-01-01", periods=n, freq="D"),
            }
        )
        expander = DateFeatureExpander()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expander.fit_transform(X, transform_dates=True)
        assert expander.expanded_indices == [0, 1]

    def test__output_names_avoid_collision_with_existing_columns(self) -> None:
        """A pre-existing column can happen to look like a generated output name."""
        X = pd.DataFrame(
            {
                "signed_on": pd.date_range("2020-01-01", periods=N, freq="D"),
                "signed_on_year": np.zeros(N),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, names, *_ = DateFeatureExpander().fit_transform(X, transform_dates=True)

        assert len(names) == len(set(names))
        # The pre-existing column keeps its name; the newly generated one deduped.
        assert names[0] == "signed_on_year"
        assert "signed_on_year_1" in names

    def test__categorical_features_indices__is_remapped_around_expansion(self) -> None:
        """A kept column after an expanded date shifts down by however many
        raw columns preceded it and were removed -- kept columns come first
        in the resolved output, in their relative order, with every expanded
        column's output appended after them.
        """
        X = pd.DataFrame(
            {
                "signed_on": pd.date_range("2020-01-01", periods=N, freq="D"),
                "cat": np.arange(N) % 3,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, _, cat, _, _ = DateFeatureExpander().fit_transform(
                X, transform_dates=True, categorical_features_indices=[1]
            )
        # "signed_on" (index 0) expanded away; "cat" is the only kept column,
        # so it becomes index 0 in the resolved output.
        assert cat == [0]

    def test__declared_categorical_date__is_never_expanded(self) -> None:
        X = _numeric_and_date_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            X_out, _, cat, num_hint, text_hint = DateFeatureExpander().fit_transform(
                X, transform_dates=True, categorical_features_indices=[1]
            )
        assert cat == [1]
        assert num_hint == []
        assert text_hint == []
        assert X_out.shape == X.shape

    def test__predict__reapplies_fitted_encoder_positionally(self) -> None:
        X_fit = _numeric_and_date_frame()
        expander = DateFeatureExpander()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            X_fit_out, *_ = expander.fit_transform(X_fit, transform_dates=True)

        X_test = _numeric_and_date_frame()
        X_test_out = expander.transform(X_test)

        assert X_test_out.shape == X_fit_out.shape
        np.testing.assert_array_equal(
            X_test_out.to_numpy(dtype=float), X_fit_out.to_numpy(dtype=float)
        )

    def test__predict__no_longer_a_date_at_predict_time__becomes_nan(self) -> None:
        """A predict-time column can drift dtype like any other fitted
        column, and there is no re-detection to fall back on. A fitted
        column no longer a genuine datetime dtype degrades to NaN, the same
        as any other missing value, rather than a best-effort parse of
        whatever is actually sitting there.
        """
        X_fit = _numeric_and_date_frame()
        expander = DateFeatureExpander()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expander.fit_transform(X_fit, transform_dates=True)

        # Same columns as at fit time; "signed_on" (index 1) has drifted to a
        # plain float dtype by predict time.
        X_test = pd.DataFrame(
            {"num": np.arange(N, dtype=float), "signed_on": np.arange(N, dtype=float)}
        )
        X_test_out = expander.transform(X_test)

        assert np.isnan(X_test_out.iloc[:, 1:].to_numpy(dtype=float)).all()

    def test__predict__a_missing_row__only_that_row_becomes_nan(self) -> None:
        X_fit = _numeric_and_date_frame()
        expander = DateFeatureExpander()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expander.fit_transform(X_fit, transform_dates=True)

        dates_with_a_gap = pd.Series(pd.date_range("2020-01-01", periods=N, freq="D"))
        dates_with_a_gap.iloc[3] = pd.NaT
        X_test = pd.DataFrame(
            {"num": np.arange(N, dtype=float), "signed_on": dates_with_a_gap}
        )
        X_test_out = expander.transform(X_test)

        assert np.isnan(X_test_out.iloc[3, 1:].to_numpy(dtype=float)).all()
        other_rows = X_test_out.drop(index=3).iloc[:, 1:]
        assert np.isfinite(other_rows.to_numpy(dtype=float)).all()


class TestWarnOnDates:
    """Unit tests for `_warn_on_dates`."""

    def test__names_given__warn_with_column_names_and_remedies(self) -> None:
        with pytest.warns(UserWarning, match="hold dates") as record:
            _warn_on_dates(["signed_on"])
        message = str(record[0].message)
        assert "'signed_on'" in message
        assert "TRANSFORM_DATES" in message
        assert "categorical_features_indices" in message

    def test__no_names__does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_on_dates([])


def test__no_double_warning_for_a_column_reported_by_the_date_warning() -> None:
    """The date warning fires; the free-text warning (fired later, by
    `detect_feature_modalities`, on the rendered text) must not repeat it.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = rng.integers(0, 2, size=n)

    clf = TabPFNClassifier(n_estimators=1, device="cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X, y)
    date_warnings = [w for w in caught if "hold dates" in str(w.message)]
    text_warnings = [w for w in caught if "look like free text" in str(w.message)]
    assert len(date_warnings) == 1
    assert not text_warnings


def test__warning_stacklevel__points_at_the_fit_call_site() -> None:
    n = 30
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n),
        }
    )
    y = rng.integers(0, 2, size=n)

    clf = TabPFNClassifier(n_estimators=1, device="cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X, y)  # this exact line must be where the warning is attributed
    date_warnings = [w for w in caught if "hold dates" in str(w.message)]
    assert len(date_warnings) == 1
    assert date_warnings[0].filename == __file__


def test__transform_dates__low_cardinality_calendar_feature_stays_numerical() -> None:
    """A calendar feature spanning few distinct values (e.g. a `year` in a
    dataset covering only two years) must stay `NUMERICAL`, never demoted to
    `CATEGORICAL` by the generic cardinality heuristic.
    """
    n = 60
    rng = np.random.default_rng(0)
    dates = pd.to_datetime(
        ["2020-06-15" if i % 2 == 0 else "2021-06-15" for i in range(n)]
    )
    X = pd.DataFrame({"num": rng.normal(size=n), "signed_on": dates})
    y = rng.integers(0, 2, size=n)

    clf = TabPFNClassifier(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    clf.fit(X, y)

    year_features = [
        f for f in clf.inferred_feature_schema_.features if "year" in f.name
    ]
    assert year_features
    assert all(f.modality is FeatureModality.NUMERICAL for f in year_features)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_predict__native_datetime_column_with_missing_value__does_not_crash(
    estimator_cls: type,
) -> None:
    """Regression: a genuine `datetime64` column with a `NaT` must not crash
    `fit`/`predict` under `TRANSFORM_DATES=True` -- the missing row's calendar
    features degrade to `NaN`, like any other missing value.
    """
    n = 60
    rng = np.random.default_rng(0)
    dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="D"))
    dates.iloc[3] = pd.NaT
    X = pd.DataFrame({"num": rng.normal(size=n), "signed_on": dates})
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X, y)
        out = (
            model.predict_proba(X)
            if estimator_cls is TabPFNClassifier
            else model.predict(X)
        )

    assert np.isfinite(out).all()


def _classification_or_regression_target(
    estimator_cls: type, rng: np.random.Generator, n: int
) -> np.ndarray:
    if estimator_cls is TabPFNClassifier:
        return rng.integers(0, 2, size=n)
    return rng.normal(size=n)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_predict__transform_dates__expands_date_and_predicts(
    estimator_cls: type,
) -> None:
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X, y)

    assert 1 in model.date_expander_.expanded_indices  # "signed_on" is 2nd column

    if estimator_cls is TabPFNClassifier:
        out = model.predict_proba(X)
    else:
        out = model.predict(X)
    assert np.isfinite(out).all()


def test__fit__declared_categorical_date__transform_dates_has_no_effect() -> None:
    """A date column declared categorical must stay excluded from
    `DatetimeEncoder` -- `TRANSFORM_DATES` must not override that intent.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = rng.integers(0, 2, size=n)

    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=[1],
        inference_config={"TRANSFORM_DATES": True},
    )
    model.fit(X, y)

    assert model.date_expander_.expanded_indices == []
    out = model.predict_proba(X)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__predict__date_expander_attribute_missing__does_not_crash(
    estimator_cls: type,
) -> None:
    """A path that skips `_initialize_dataset_preprocessing` (e.g.
    `fit_from_preprocessed`) never sets `date_expander_` at all -- predict
    must not crash on that, the same way it already tolerates a missing
    `ordinal_encoder_`.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"num": rng.normal(size=n)})
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(n_estimators=1, device="cpu")
    model.fit(X, y)
    del model.date_expander_

    if estimator_cls is TabPFNClassifier:
        out = model.predict_proba(X)
    else:
        out = model.predict(X)
    assert np.isfinite(out).all()


def test__predict_proba_batched__transform_dates__reapplies_encoder_on_worker() -> None:
    """The ensemble-worker predict path also reapplies the fitted date encoder,
    not just the direct-`self` path.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = rng.integers(0, 2, size=n)

    clf = TabPFNClassifier(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    proba = clf.predict_proba_batched([X], [y], [X[:5]])
    assert proba.shape == (1, 5, 2)
    assert np.isfinite(proba).all()


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__get_embeddings__transform_dates__expands_before_the_ordinal_encoder(
    estimator_cls: type,
) -> None:
    """`get_embeddings` has its own predict-input path, separate from
    `predict`/`predict_proba` -- it must also expand dates first, or the
    ordinal encoder sees a column count it was never fitted with.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )
    y = _classification_or_regression_target(estimator_cls, rng, n)

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_DATES": True}
    )
    model.fit(X, y)

    embeddings = get_embeddings(model, X, data_source="test")
    assert np.isfinite(embeddings).all()


def test__get_embeddings__transform_dates__categorical_indices_shift_with_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A date column expanding *before* a declared-categorical one shifts that
    column's position. `get_embeddings` must pass the post-expansion index to
    `fix_dtypes`, not the raw, pre-expansion `categorical_features_indices` --
    otherwise it silently marks the wrong (date-derived, numeric) column as
    categorical instead, with no error to reveal it.
    """
    n = 60
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "signed_on": pd.date_range("2020-01-01", periods=n, freq="D"),
            "num": rng.normal(size=n),
            "cat": rng.integers(0, 3, size=n),
        }
    )
    y = rng.integers(0, 2, size=n)

    model = TabPFNClassifier(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=[2],  # "cat", before expansion
        inference_config={"TRANSFORM_DATES": True},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    # The date column expanded and was moved to the end, so "cat" shifted from
    # raw index 2 down to 1 -- distinct from the stale raw index.
    assert model.categorical_features_indices == [2]
    assert model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL) == [
        1
    ]

    seen_cat_indices = []
    original_fix_dtypes = tabpfn.base.fix_dtypes

    def _spy_fix_dtypes(X, cat_indices, **kwargs) -> pd.DataFrame:
        seen_cat_indices.append(cat_indices)
        return original_fix_dtypes(X, cat_indices, **kwargs)

    monkeypatch.setattr(tabpfn.base, "fix_dtypes", _spy_fix_dtypes)

    get_embeddings(model, X, data_source="test")

    assert seen_cat_indices == [[1]]
