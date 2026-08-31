#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the opt-in built-model cache in ``tabpfn.model_loading``."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from tabpfn import model_loading


@pytest.fixture
def ckpt(tmp_path: Path) -> Path:
    # Only stat() is read (Checkpoint.identity); the build itself is patched.
    p = tmp_path / "model.ckpt"
    p.write_bytes(b"not a real checkpoint")
    return p


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    model_loading.clear_built_model_cache()
    yield
    model_loading.clear_built_model_cache()


def _patch_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace the real build with a counter returning a fresh sentinel tuple."""
    calls = {"n": 0}

    def fake_build(*_args: object, **_kwargs: object) -> tuple:
        calls["n"] += 1
        return (object(), None, object(), object())

    monkeypatch.setattr(model_loading, "_build_model", fake_build)
    return calls


def test_cache_hit_reuses_built_model(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    first = model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    second = model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )

    assert first is second  # same built model handed back
    assert calls["n"] == 1  # built once, not twice


def test_one_build_serves_both_fit_modes(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    """`cache_trainset_representation` is not in the key, so both modes share.

    Every architecture ignores the flag, so the two builds are identical. Giving
    it its own key would only double the entries — and would be actively wrong
    for an architecture that did honour it, since such a model accumulates
    per-fit state and must not be shared at all.
    """
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    for cache_trainset_representation in (False, True, False, True):
        model_loading.load_model(
            path=ckpt,
            estimator_type="classifier",
            cache_trainset_representation=cache_trainset_representation,
        )
    assert calls["n"] == 1


def test_cache_enabled_by_default(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.delenv("TABPFN_MODEL_CACHE_SIZE", raising=False)

    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    assert calls["n"] == 1


def test_default_size_holds_a_classifier_and_a_regressor(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    """The default of 2 is chosen so neither task evicts the other."""
    calls = _patch_build(monkeypatch)
    monkeypatch.delenv("TABPFN_MODEL_CACHE_SIZE", raising=False)

    for estimator_type in ("classifier", "regressor", "classifier", "regressor"):
        model_loading.load_model(
            path=ckpt,
            estimator_type=estimator_type,
            cache_trainset_representation=False,
        )
    assert calls["n"] == 2


def test_cache_can_be_disabled(ckpt: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "0")

    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    assert calls["n"] == 2


def test_invalid_size_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "not-a-number")
    assert (
        model_loading._get_built_model_cache_size()
        == model_loading._DEFAULT_BUILT_MODEL_CACHE_SIZE
    )


def test_models_for_different_devices_are_not_shared(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    """The caller moves the shared instance in place, so devices must be keyed.

    Without the device in the key, a fit on one device hands back an instance
    another device's estimator had already moved, and predict blows up.
    """
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    cpu = model_loading.load_model(
        path=ckpt,
        estimator_type="classifier",
        cache_trainset_representation=False,
        devices=[torch.device("cpu")],
    )
    meta = model_loading.load_model(
        path=ckpt,
        estimator_type="classifier",
        cache_trainset_representation=False,
        devices=[torch.device("meta")],
    )
    cpu_again = model_loading.load_model(
        path=ckpt,
        estimator_type="classifier",
        cache_trainset_representation=False,
        devices=[torch.device("cpu")],
    )

    assert cpu is not meta
    assert cpu is cpu_again
    assert calls["n"] == 2


def test_models_for_different_dtypes_are_not_shared(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    """The caller casts the shared instance in place, so the dtype must be keyed.

    Without it, a half-precision fit leaves the cached model in fp16 and the next
    full-precision fit fails with "mat1 and mat2 must have the same dtype".
    """
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    def load(dtype: torch.dtype | None) -> tuple:
        return model_loading.load_model(
            path=ckpt,
            estimator_type="classifier",
            cache_trainset_representation=False,
            devices=[torch.device("cpu")],
            force_inference_dtype=dtype,
        )

    full = load(None)
    half = load(torch.float16)
    assert full is not half
    assert full is load(None)
    assert half is load(torch.float16)
    assert calls["n"] == 2


def test_unspecified_devices_are_kept_apart_from_device_specific_entries(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    model_loading.load_model(
        path=ckpt,
        estimator_type="classifier",
        cache_trainset_representation=False,
        devices=[torch.device("cpu")],
    )
    assert calls["n"] == 2


def test_clear_built_model_cache_forces_a_rebuild(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    """`TabPFN*.to()` clears the cache before re-placing what it holds."""
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")

    def load() -> tuple:
        return model_loading.load_model(
            path=ckpt,
            estimator_type="classifier",
            cache_trainset_representation=False,
            devices=[torch.device("cpu")],
        )

    first = load()
    model_loading.clear_built_model_cache()
    second = load()

    assert first is not second
    assert calls["n"] == 2


def test_caching_a_build_releases_the_raw_checkpoint(
    ckpt: Path, monkeypatch: pytest.MonkeyPatch
):
    """The built model is a full copy of the weights; keeping both doubles memory."""
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "4")
    monkeypatch.setattr(
        model_loading,
        "_build_model",
        lambda *_a, **_k: (object(), None, object(), object()),
    )
    # The fixture is not a real checkpoint, so stub the read out and prime the
    # raw-checkpoint cache by hand.
    monkeypatch.setattr(model_loading.Checkpoint, "load", lambda _self: {})
    model_loading._load_checkpoint_cached.cache_clear()
    resolved = str(ckpt.resolve())
    model_loading._load_checkpoint_cached(
        resolved, model_loading.Checkpoint(resolved).identity()
    )
    assert model_loading._load_checkpoint_cached.cache_info().currsize == 1

    model_loading.load_model(
        path=ckpt, estimator_type="classifier", cache_trainset_representation=False
    )
    assert model_loading._load_checkpoint_cached.cache_info().currsize == 0


def test_lru_eviction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = _patch_build(monkeypatch)
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "1")
    a = tmp_path / "a.ckpt"
    a.write_bytes(b"a")
    b = tmp_path / "b.ckpt"
    b.write_bytes(b"b")

    model_loading.load_model(
        path=a, estimator_type="classifier", cache_trainset_representation=False
    )
    model_loading.load_model(
        path=b, estimator_type="classifier", cache_trainset_representation=False
    )  # evicts a
    model_loading.load_model(
        path=a, estimator_type="classifier", cache_trainset_representation=False
    )  # rebuilds a
    assert calls["n"] == 3


def test_load_model_signature_is_tracked_by_the_cache():
    """Tripwire: the cache correctness depends on `load_model`'s exact inputs.

    It keys on (path, file-identity, estimator type, placement) and caches only when
    ``cache_trainset_representation`` is False. So a *new build-affecting*
    parameter must be added to the cache key (else a hit returns a stale model),
    and a *new mutation flag* must extend the gate (else a mutated model gets
    shared). If this assertion fails, revisit `_BUILT_MODEL_CACHE` / `load_model`
    before updating the expected set.
    """
    params = set(inspect.signature(model_loading.load_model).parameters)
    assert params == {
        "path",
        "estimator_type",
        "cache_trainset_representation",
        "devices",
        "force_inference_dtype",
    }, (
        f"load_model parameters changed to {sorted(params)}; the built-model "
        "cache key and/or its cache_trainset_representation gate must be updated."
    )
