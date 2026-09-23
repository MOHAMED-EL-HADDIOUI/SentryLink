import dataclasses

import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from sentrylink.crypto.secure_aggregation import (
    SecAggClient,
    SecAggServer,
    clip_l2,
    dequantize,
    new_round_id,
    quantize,
)
from sentrylink.config import UPDATE_CLIP


def _make_round(roster, dim, round_id=None):
    round_id = round_id or new_round_id()
    server = SecAggServer(round_id=round_id, roster=roster, dim=dim)
    clients = {oid: SecAggClient(oid, round_id, roster, dim) for oid in roster}
    for oid, c in clients.items():
        server.register(oid, c.public_key)
    return server, clients


def _contains_privkey(obj, _seen=None):
    """True if any X25519 private key is reachable from obj."""
    _seen = _seen if _seen is not None else set()
    if id(obj) in _seen:
        return False
    _seen.add(id(obj))
    if isinstance(obj, X25519PrivateKey):
        return True
    if isinstance(obj, dict):
        return any(_contains_privkey(v, _seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return any(_contains_privkey(v, _seen) for v in obj)
    if hasattr(obj, "__dataclass_fields__"):
        return any(
            _contains_privkey(getattr(obj, f.name), _seen)
            for f in dataclasses.fields(obj)
        )
    if hasattr(obj, "__dict__"):
        return _contains_privkey(vars(obj), _seen)
    return False


def test_quantize_roundtrip():
    v = np.array([0.1, -0.25, 3.5])
    q = quantize(v)
    assert q.dtype == np.int64
    np.testing.assert_allclose(dequantize(q), v, atol=1e-6)


def test_clip_l2():
    v = np.array([3.0, 4.0])  # norm 5
    c = clip_l2(v, bound=1.0)
    assert abs(np.linalg.norm(c) - 1.0) <= 1e-6


def test_masks_cancel_without_dropout():
    roster = ["a", "b", "c"]
    dim = 8
    server, clients = _make_round(roster, dim)
    rng = np.random.default_rng(0)
    updates = {oid: rng.normal(size=dim) for oid in roster}
    keys = server.public_keys()
    for oid, u in updates.items():
        server.receive(clients[oid].mask_and_send(u, keys))

    got = server.finalize()
    expected = dequantize(sum((quantize(clip_l2(u)) for u in updates.values()), np.zeros(dim, dtype=np.int64)))
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_masks_cancel_with_dropout():
    roster = ["a", "b", "c", "d"]
    dim = 16
    server, clients = _make_round(roster, dim)
    rng = np.random.default_rng(1)
    updates = {oid: rng.normal(size=dim) for oid in roster}
    keys = server.public_keys()
    active = ["a", "b", "c"]
    for oid in active:
        server.receive(clients[oid].mask_and_send(updates[oid], keys))
    for survivor in active:
        server.add_recovery_seed(
            survivor, "d", clients[survivor].reveal_seed("d", keys["d"])
        )

    got = server.finalize()
    expected_vec = sum(
        (quantize(clip_l2(updates[oid])) for oid in active), np.zeros(dim, dtype=np.int64)
    )
    np.testing.assert_allclose(got, dequantize(expected_vec), atol=1e-9)


def test_updates_clipped_before_masking():
    roster = ["a", "b"]
    server, clients = _make_round(roster, 4)
    keys = server.public_keys()
    big = np.full(4, 100.0)
    server.receive(clients["a"].mask_and_send(big, keys))
    server.receive(clients["b"].mask_and_send(np.zeros(4), keys))
    got = server.finalize()
    clipped = clip_l2(big, UPDATE_CLIP)
    np.testing.assert_allclose(got, clipped, atol=1e-6)


def test_server_holds_no_private_key_material():
    """Trust boundary, enforced: private keys live client-side only."""
    roster = ["a", "b", "c"]
    server, clients = _make_round(roster, 4)
    assert all(
        isinstance(c._private_key, X25519PrivateKey) for c in clients.values()
    )
    rng = np.random.default_rng(3)
    keys = server.public_keys()
    for oid, c in clients.items():
        update = rng.normal(size=4)
        contrib = c.mask_and_send(update, keys)
        # masks are actually applied: the server's view differs from our update
        assert not np.array_equal(contrib.masked_update, quantize(clip_l2(update)))
        server.receive(contrib)
    server.add_recovery_seed("a", "b", clients["a"].reveal_seed("b", keys["b"]))
    assert not _contains_privkey(server)


def _walk_values(obj, _seen=None):
    """Yield every reachable scalar/bytes value (for byte-level scans)."""
    _seen = _seen if _seen is not None else set()
    if id(obj) in _seen:
        return
    _seen.add(id(obj))
    if isinstance(obj, (bytes, bytearray)):
        yield bytes(obj)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_values(v, _seen)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            yield from _walk_values(v, _seen)
        return
    if hasattr(obj, "__dataclass_fields__"):
        for f in dataclasses.fields(obj):
            yield from _walk_values(getattr(obj, f.name), _seen)
        return
    if hasattr(obj, "__dict__"):
        yield from _walk_values(vars(obj), _seen)


def test_server_holds_no_private_key_bytes_or_seeds():
    roster = ["a", "b", "c"]
    server, clients = _make_round(roster, 4)
    rng = np.random.default_rng(11)
    keys = server.public_keys()
    for oid, c in clients.items():
        server.receive(c.mask_and_send(rng.normal(size=4), keys))
    server.finalize()
    assert server._revealed_seeds == {}  # no dropout: no seed retained
    private_raw = {c._private_key.private_bytes_raw() for c in clients.values()}
    blob = b"".join(v for v in _walk_values(server) if isinstance(v, bytes))
    for raw in private_raw:
        assert raw not in blob
    assert all(raw not in repr(server).encode() for raw in private_raw)


def test_server_has_no_reference_path_to_clients():
    roster = ["a", "b"]
    server, clients = _make_round(roster, 4)
    keys = server.public_keys()
    for oid, c in clients.items():
        server.receive(c.mask_and_send(np.zeros(4), keys))
    assert not _contains_privkey(server)

    found = []

    def _find_clients(obj, _seen=None):
        _seen = _seen if _seen is not None else set()
        if id(obj) in _seen:
            return
        _seen.add(id(obj))
        if isinstance(obj, SecAggClient):
            found.append(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _find_clients(v, _seen)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for v in obj:
                _find_clients(v, _seen)
        elif hasattr(obj, "__dataclass_fields__"):
            for f in dataclasses.fields(obj):
                _find_clients(getattr(obj, f.name), _seen)
        elif hasattr(obj, "__dict__"):
            _find_clients(vars(obj), _seen)

    _find_clients(server)
    assert found == []


def test_server_shape_has_no_hidden_key_fields():
    # Tripwire: adding a private-key holder to SecAggServer breaks this on
    # purpose, forcing the author to update the isolation tests too.
    assert {f.name for f in dataclasses.fields(SecAggServer)} == {
        "round_id", "roster", "dim", "_keys", "_contributions", "_revealed_seeds",
    }


def test_server_survives_copy_without_leaking():
    import copy

    roster = ["a", "b"]
    server, clients = _make_round(roster, 4)
    keys = server.public_keys()
    for oid, c in clients.items():
        server.receive(c.mask_and_send(np.ones(4), keys))
    clone = copy.deepcopy(server)
    assert not _contains_privkey(clone)
    np.testing.assert_allclose(clone.finalize(), server.finalize(), atol=1e-12)


def test_server_rejects_non_roster_and_bad_dims():
    server, clients = _make_round(["a", "b"], 4)
    import pytest

    with pytest.raises(PermissionError):
        server.register("mallory", clients["a"].public_key)
    with pytest.raises(PermissionError):
        SecAggClient("mallory", server.round_id, ["a", "b"], 4)
    with pytest.raises(ValueError):
        clients["a"].mask_and_send(np.zeros(7), server.public_keys())
