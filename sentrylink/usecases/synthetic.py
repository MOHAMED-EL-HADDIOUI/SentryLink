"""Synthetic multi-organization datasets for the three showcase verticals.

Each generator returns per-org local datasets that simulate siloed data:
no organization ever sees another's rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OrgDataset:
    org_id: str
    name: str
    x: np.ndarray
    y: np.ndarray
    extra: dict


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_retail_cohort(
    n_orgs: int = 5,
    n_per_org: int = 400,
    seed: int = 7,
) -> list[OrgDataset]:
    """Retailers: predict demand lift (1) from promo depth, price index,
    weather index, foot-traffic index — an emerging cross-region pattern."""
    rng = np.random.default_rng(seed)
    orgs = []
    true_w = np.array([1.4, -1.1, 0.9, 0.7])
    for i in range(n_orgs):
        region_shift = rng.normal(0, 0.15, size=true_w.shape)
        x = rng.normal(0, 1, size=(n_per_org, 4))
        # simulate a shared emerging trend: promo effect strengthens
        logit = x @ (true_w + region_shift) + rng.normal(0, 0.3, size=n_per_org)
        y = (rng.uniform(size=n_per_org) < _sigmoid(logit)).astype(np.float64)
        orgs.append(
            OrgDataset(
                org_id=f"ret{i}",
                name=f"Retailer {i + 1}",
                x=x,
                y=y,
                extra={"region": chr(ord("A") + i)},
            )
        )
    return orgs


def make_manufacturing_cohort(
    n_orgs: int = 4,
    n_per_org: int = 350,
    seed: int = 11,
) -> list[OrgDataset]:
    """Plants: predict quality defect (1) from vibration, temperature,
    humidity, tool-age — a shared failure mode across the supply chain."""
    rng = np.random.default_rng(seed)
    orgs = []
    true_w = np.array([1.2, 1.0, 0.5, 0.8])
    for i in range(n_orgs):
        tool_base = rng.normal(0, 0.2, size=true_w.shape)
        x = rng.normal(0, 1, size=(n_per_org, 4))
        logit = x @ (true_w + tool_base) + rng.normal(0, 0.35, size=n_per_org)
        y = (rng.uniform(size=n_per_org) < _sigmoid(logit)).astype(np.float64)
        orgs.append(
            OrgDataset(
                org_id=f"mfg{i}",
                name=f"Plant {i + 1}",
                x=x,
                y=y,
                extra={"line": i % 3},
            )
        )
    return orgs


def make_healthcare_cohort(
    n_orgs: int = 4,
    n_per_org: int = 300,
    seed: int = 23,
) -> list[OrgDataset]:
    """Hospitals: predict positive treatment response from age band,
    baseline severity, biomarker, treatment arm — response clusters."""
    rng = np.random.default_rng(seed)
    orgs = []
    true_w = np.array([0.4, -1.3, 1.1, 0.6])
    for i in range(n_orgs):
        site_cal = rng.normal(0, 0.12, size=true_w.shape)
        x = rng.normal(0, 1, size=(n_per_org, 4))
        logit = x @ (true_w + site_cal) + rng.normal(0, 0.4, size=n_per_org)
        y = (rng.uniform(size=n_per_org) < _sigmoid(logit)).astype(np.float64)
        orgs.append(
            OrgDataset(
                org_id=f"hosp{i}",
                name=f"Hospital {i + 1}",
                x=x,
                y=y,
                extra={"site": f"S{i + 1}"},
            )
        )
    return orgs
