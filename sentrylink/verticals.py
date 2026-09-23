"""Vertical (domain) policies and synthetic dataset descriptors.

Core governance consumes these objects instead of hard-coded branches, so a
new vertical is data, not a code fork. The built-in policies reproduce the
historical defaults exactly (covered by tests/test_verticals.py).

Dataset classes wrap the deterministic synthetic generators in
usecases/synthetic.py with documented feature meanings. All data is
synthetic; healthcare features are explicitly non-identifying.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .usecases.synthetic import (
    OrgDataset,
    make_healthcare_cohort,
    make_manufacturing_cohort,
    make_retail_cohort,
)


class DatasetSpec(ABC):
    DESCRIPTION: str
    FEATURES: tuple[tuple[str, str], ...]
    LABEL: str

    @classmethod
    @abstractmethod
    def generate(
        cls, n_orgs: int = 4, n_per_org: int = 100, seed: int = 0
    ) -> list[OrgDataset]:
        raise NotImplementedError


@dataclass(frozen=True)
class VerticalPolicy:
    key: str  # domain string, e.g. "healthcare"
    title: str
    description: str
    min_participants: int
    allowed_metrics: frozenset[str]
    max_epsilon_per_query: float
    purpose: str


RETAIL = VerticalPolicy(
    key="retail",
    title="Retail",
    description="Retailers pooling emerging demand patterns across regions.",
    min_participants=3,
    allowed_metrics=frozenset(
        {"histogram", "variance", "correlation", "federated_model_round"}
    ),
    max_epsilon_per_query=25.0,
    purpose="cross-org intelligence",
)

MANUFACTURING = VerticalPolicy(
    key="manufacturing",
    title="Manufacturing",
    description="Plants pooling quality/defect signals across the supply chain.",
    min_participants=3,
    allowed_metrics=frozenset(
        {"histogram", "variance", "correlation", "federated_model_round"}
    ),
    max_epsilon_per_query=25.0,
    purpose="cross-org intelligence",
)

HEALTHCARE = VerticalPolicy(
    key="healthcare",
    title="Healthcare",
    description=(
        "Hospitals pooling treatment-response clusters. Stricter by design: "
        "higher cohort floor, no variance metric, lower per-query epsilon."
    ),
    min_participants=4,
    allowed_metrics=frozenset({"histogram", "correlation", "federated_model_round"}),
    max_epsilon_per_query=10.0,
    purpose="treatment-response research",
)

DEFAULT_VERTICAL = VerticalPolicy(
    key="default",
    title="Default",
    description="Fallback policy for domains without a dedicated vertical.",
    min_participants=3,
    allowed_metrics=frozenset(
        {"histogram", "variance", "correlation", "federated_model_round"}
    ),
    max_epsilon_per_query=25.0,
    purpose="cross-org intelligence",
)

VERTICALS: dict[str, VerticalPolicy] = {
    "retail": RETAIL,
    "manufacturing": MANUFACTURING,
    "healthcare": HEALTHCARE,
}


def policy_for_domain(domain: str) -> VerticalPolicy:
    return VERTICALS.get(domain, DEFAULT_VERTICAL)


class RetailDataset(DatasetSpec):
    """Synthetic retailer data: demand lift from promo/price/weather/traffic."""

    DESCRIPTION = (
        "Predict demand lift (1) from promo depth, price index, weather index "
        "and foot-traffic index, with an emerging cross-region promo trend."
    )
    FEATURES = (
        ("promo_depth", "promotion depth index"),
        ("price_index", "relative price index"),
        ("weather_index", "weather favorability index"),
        ("foot_traffic", "store foot-traffic index"),
    )
    LABEL = "demand lift (1 = lifted, 0 = baseline)"

    @classmethod
    def generate(
        cls, n_orgs: int = 5, n_per_org: int = 400, seed: int = 7
    ) -> list[OrgDataset]:
        return make_retail_cohort(n_orgs=n_orgs, n_per_org=n_per_org, seed=seed)


class ManufacturingDataset(DatasetSpec):
    """Synthetic plant data: quality defects from sensor/tooling features."""

    DESCRIPTION = (
        "Predict quality defects (1) from vibration, temperature, humidity and "
        "tool age, with a shared failure mode across the supply chain."
    )
    FEATURES = (
        ("vibration", "vibration severity index"),
        ("temperature", "process temperature index"),
        ("humidity", "ambient humidity index"),
        ("tool_age", "tool-age index"),
    )
    LABEL = "quality defect (1 = defect, 0 = ok)"

    @classmethod
    def generate(
        cls, n_orgs: int = 4, n_per_org: int = 350, seed: int = 11
    ) -> list[OrgDataset]:
        return make_manufacturing_cohort(n_orgs=n_orgs, n_per_org=n_per_org, seed=seed)


class HealthcareDataset(DatasetSpec):
    """Synthetic hospital data: treatment response clusters.

    Explicitly non-identifying: age bands, severity bands, a biomarker index
    and a treatment-arm flag. No real personal or medical records, ever.
    """

    DESCRIPTION = (
        "Predict positive treatment response from age band, baseline severity, "
        "biomarker index and treatment arm, with site-level calibration."
    )
    FEATURES = (
        ("age_band", "patient age band index (synthetic, non-identifying)"),
        ("baseline_severity", "baseline severity index"),
        ("biomarker", "biomarker index"),
        ("treatment_arm", "treatment arm flag"),
    )
    LABEL = "positive treatment response (1 = response, 0 = none)"

    @classmethod
    def generate(
        cls, n_orgs: int = 4, n_per_org: int = 300, seed: int = 23
    ) -> list[OrgDataset]:
        return make_healthcare_cohort(n_orgs=n_orgs, n_per_org=n_per_org, seed=seed)


DATASETS: dict[str, type[DatasetSpec]] = {
    "retail": RetailDataset,
    "manufacturing": ManufacturingDataset,
    "healthcare": HealthcareDataset,
}
