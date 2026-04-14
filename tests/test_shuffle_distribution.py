import numpy as np
import pytest
from meval.config import settings
from meval.stats import RandomState, shuffle_masks, shuffle_masks_from_state

def _legacy_shuffle_from_state(idces_joined: np.ndarray, n_a: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idces_permuted = rng.permutation(idces_joined)
    return idces_permuted[:n_a], idces_permuted[n_a:]

def test_shuffle_masks_partition_properties():
    old_settings = settings.to_dict().copy()
    try:
        settings.update(debug=True)
        RandomState.reset()

        # Use array size > 128 to actually trigger the Numba logic
        idces_joined = np.arange(130, dtype=np.int64)
        n_a = 40

        # work_buffer is required to trigger Numba path
        work = np.empty_like(idces_joined)

        a, b = shuffle_masks(
            idces_joined=idces_joined,
            n_a=n_a,
            work_buffer=work,
        )

        assert len(a) == n_a
        assert len(b) == len(idces_joined) - n_a
        assert len(np.intersect1d(a, b)) == 0
        assert np.array_equal(np.sort(np.concatenate([a, b])), idces_joined)
    finally:
        settings.from_dict(old_settings)


def test_shuffle_masks_debug_assert_catches_invalid_partition_size():
    old_settings = settings.to_dict().copy()
    try:
        settings.update(debug=True)
        idces_joined = np.arange(10, dtype=np.int64)
        work = np.empty_like(idces_joined)

        # Invalid split: n_a cannot exceed total size.
        with pytest.raises(AssertionError):
            shuffle_masks_from_state(
                idces_joined=idces_joined,
                n_a=11,
                work_buffer=work,
                rng=np.random.default_rng(0),
            )
    finally:
        settings.from_dict(old_settings)

def test_shuffle_distribution_matches_legacy_marginals():
    # Force the Numba path by using a size > 128
    idces_joined = np.arange(130, dtype=np.int64)
    n_a = 50
    draws = 20000  # High enough to keep statistical variance tight

    rng_new = np.random.default_rng(123)
    rng_old = np.random.default_rng(123)

    counts_new = np.zeros(len(idces_joined), dtype=np.int64)
    counts_old = np.zeros(len(idces_joined), dtype=np.int64)

    work = np.empty_like(idces_joined)

    for _ in range(draws):
        a_new, _ = shuffle_masks_from_state(
            idces_joined=idces_joined,
            n_a=n_a,
            work_buffer=work,
            rng=rng_new,
        )
        a_old, _ = _legacy_shuffle_from_state(idces_joined=idces_joined, n_a=n_a, rng=rng_old)

        counts_new[a_new] += 1
        counts_old[a_old] += 1

    p_new = counts_new / draws
    p_old = counts_old / draws

    expected = n_a / len(idces_joined)

    # Marginals should be close to uniform for both methods.
    # Standard deviation for p=0.38, N=20000 is ~0.0034.
    # A tolerance of 0.015 gives roughly a generous 4-sigma bound to prevent flaky tests.
    assert np.max(np.abs(p_new - expected)) < 0.015
    assert np.max(np.abs(p_old - expected)) < 0.015

    # New implementation should match the legacy baseline distributionally.
    assert np.max(np.abs(p_new - p_old)) < 0.015