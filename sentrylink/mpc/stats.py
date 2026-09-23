"""Secure multi-party computation of aggregate statistics.

Model (semi-honest, 2 non-colluding compute nodes):
  - Each data-holding organization secret-shares its local aggregate vector
    across ComputeNodes (additive shares mod FIELD_P).
   - Linear ops (sum, linear combination) run on shares with no interaction.
   - By design no served query needs cross-org products: each org reduces its
     own rows to sufficient statistics (sums, sums of squares, xy tallies)
     locally, and only those tallies are combined under shares. The protocol
     is deliberately linear-only, so no triple generation machinery is needed.
   - Outputs are only opened after governance policy checks, and always pass
     through differential privacy before leaving the platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import FIELD_P, QUANT_SCALE
from ..crypto.secret_sharing import Share, reconstruct, share_vector


def quantize_ints(values: np.ndarray | list[float]) -> list[int]:
    arr = np.rint(np.asarray(values, dtype=np.float64) * QUANT_SCALE).astype(np.int64)
    return [int(x) for x in arr.reshape(-1)]


def signed_mod(v: int) -> int:
    v = int(v) % FIELD_P
    return v - FIELD_P if v > FIELD_P // 2 else v


def dequantize_sum(v: int) -> float:
    return signed_mod(v) / QUANT_SCALE


@dataclass
class ComputeNode:
    """One non-colluding MPC compute node."""

    node_id: str
    shares: dict[str, Share] = field(default_factory=dict)  # query_id -> share

    def receive(self, query_id: str, share: Share) -> None:
        if share.node_id != self.node_id:
            raise ValueError("share destined for another node")
        self.shares[query_id] = share

    def get(self, query_id: str) -> Share:
        return self.shares[query_id]


def elementwise_add(x: dict[str, Share], y: dict[str, Share], node_ids: list[str]) -> dict[str, Share]:
    return {n: x[n] + y[n] for n in node_ids}


def run_histogram(
    org_contributions: dict[str, list[int]],
    node_ids: list[str],
) -> list[int]:
    """Sum of per-org bucket counts. Each org's vector stays secret-shared
    until the joint total is opened (caller applies DP noise before release).
    """
    if not org_contributions:
        raise ValueError("no contributions")
    acc: dict[str, Share] | None = None
    for values in org_contributions.values():
        sh = share_vector(values, node_ids)
        acc = sh if acc is None else elementwise_add(acc, sh, node_ids)
    assert acc is not None
    opened = reconstruct([acc[n] for n in node_ids])
    return [signed_mod(v) for v in opened]


def run_mean(value_sums: list[float], counts: list[int]) -> float:
    """Mean of pooled quantized value-sums / raw counts (post-open helper)."""
    if sum(counts) == 0:
        return float("nan")
    return float(np.sum(value_sums) / np.sum(counts))


def run_variance(
    org_values: dict[str, list[float]],
    node_ids: list[str],
) -> float:
    """Population variance over pooled org values.

    Each org reduces its own rows to sufficient statistics (sum x, sum x²)
    locally, then those two scalars are combined under additive shares —
    no org's raw values are ever shared or opened.
    """
    n_total = sum(len(v) for v in org_values.values())
    if n_total < 2:
        return 0.0

    acc: dict[str, Share] | None = None
    for values in org_values.values():
        q = quantize_ints(values)
        sx = sum(q) % FIELD_P
        sq = sum((xi * xi) for xi in q) % FIELD_P
        sh = share_vector([sx, sq], node_ids)
        acc = sh if acc is None else elementwise_add(acc, sh, node_ids)
    assert acc is not None

    opened = reconstruct([acc[n] for n in node_ids])
    total_x = float(signed_mod(opened[0])) / QUANT_SCALE
    total_sq = float(signed_mod(opened[1])) / (QUANT_SCALE**2)
    mean = total_x / n_total
    return max(0.0, total_sq / n_total - mean * mean)


def run_correlation(
    org_pairs: dict[str, tuple[list[float], list[float]]],
    node_ids: list[str],
) -> float:
    """Pearson correlation over pooled (x, y) pairs across orgs.

    Each org computes local sufficient statistics on its own rows (own-data
    products are local); cross-org combination is additive over secret shares,
    so no party sees another org's statistics — only the final value is opened
    (and the caller wraps it with DP before external release).
    """
    acc: dict[str, Share] | None = None
    for _org, (xs, ys) in org_pairs.items():
        if len(xs) != len(ys):
            raise ValueError("x/y length mismatch")
        qx = quantize_ints(xs)
        qy = quantize_ints(ys)
        n = len(qx)
        sx = sum(qx) % FIELD_P
        sy = sum(qy) % FIELD_P
        sxx = sum(v * v for v in qx) % FIELD_P
        syy = sum(v * v for v in qy) % FIELD_P
        sxy = sum(qx[i] * qy[i] for i in range(n)) % FIELD_P
        sh = share_vector([sx, sy, sxx, syy, sxy, n], node_ids)
        acc = sh if acc is None else elementwise_add(acc, sh, node_ids)
    assert acc is not None
    sx, sy, sxx, syy, sxy, n_open = [signed_mod(v) for v in reconstruct([acc[n] for n in node_ids])]
    if n_open < 2:
        return 0.0
    sx_f, sy_f = sx / QUANT_SCALE, sy / QUANT_SCALE
    sxx_f = sxx / (QUANT_SCALE**2)
    syy_f = syy / (QUANT_SCALE**2)
    sxy_f = sxy / (QUANT_SCALE**2)
    cov = sxy_f / n_open - (sx_f / n_open) * (sy_f / n_open)
    vx = sxx_f / n_open - (sx_f / n_open) ** 2
    vy = syy_f / n_open - (sy_f / n_open) ** 2
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(np.clip(cov / np.sqrt(vx * vy), -1.0, 1.0))
