import numpy as np
import pytest

from sentrylink.crypto.prg import expand_seed, int_masks
from sentrylink.crypto.secret_sharing import reconstruct, share_scalar, share_vector
from sentrylink.config import FIELD_P, MASK_BOUND


def test_expand_seed_deterministic():
    seed = b"\x01" * 32
    assert expand_seed(seed, 64) == expand_seed(seed, 64)
    assert expand_seed(seed, 64) != expand_seed(b"\x02" * 32, 64)


def test_int_masks_range_and_determinism():
    seed = b"\xab" * 32
    m1 = int_masks(seed, 1000)
    m2 = int_masks(seed, 1000)
    assert np.array_equal(m1, m2)
    assert m1.min() >= -MASK_BOUND
    assert m1.max() <= MASK_BOUND
    assert int_masks(seed, 0).shape == (0,)


def test_share_reconstruct_roundtrip():
    values = [1, 42, 999999]
    shares = share_vector(values, ["n1", "n2", "n3"])
    assert reconstruct(list(shares.values())) == values


def test_share_two_nodes():
    values = [123456789, 987654321]
    shares = share_vector(values, ["a", "b"])
    assert reconstruct(list(shares.values())) == values


def test_share_rejects_out_of_field():
    with pytest.raises(ValueError):
        share_vector([FIELD_P + 1], ["a", "b"])


def test_share_scalar():
    shares = share_scalar(77, ["x", "y"])
    assert reconstruct(list(shares.values())) == [77]


def test_share_add_sub():
    s1 = share_vector([10, 20], ["a", "b"])
    s2 = share_vector([3, 4], ["a", "b"])
    total = {k: s1[k] + s2[k] for k in s1}
    assert reconstruct(list(total.values())) == [13, 24]
    diff = {k: s1[k] - s2[k] for k in s1}
    assert reconstruct(list(diff.values())) == [7, 16]
