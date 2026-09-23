import numpy as np
import pytest

from sentrylink.crypto.differential_privacy import (
    PrivacyAccountant,
    PrivacyBudget,
    gaussian_noise,
    gaussian_rdp_cost,
    gaussian_sigma,
    laplace_noise,
    laplace_rdp_cost,
    rdp_to_epsilon,
)


def test_sigma_increases_as_epsilon_shrinks():
    s1 = gaussian_sigma(1.0, 1.0, 1e-5)
    s2 = gaussian_sigma(1.0, 0.1, 1e-5)
    assert s2 > s1


def test_laplace_scale_and_mean():
    rng = np.random.default_rng(7)
    noise = laplace_noise(200_000, sensitivity=2.0, epsilon=8.0, rng=rng)
    # Laplace(b= sens/eps): std = b * sqrt(2)
    b = 2.0 / 8.0
    assert abs(float(np.std(noise)) - b * np.sqrt(2)) / (b * np.sqrt(2)) < 0.05
    assert abs(float(np.mean(noise))) < 0.05


def test_sigma_increases_as_delta_shrinks():
    s1 = gaussian_sigma(1.0, 1.0, 1e-3)
    s2 = gaussian_sigma(1.0, 1.0, 1e-6)
    assert s2 > s1


def test_noise_scale():
    rng = np.random.default_rng(42)
    sigma = gaussian_sigma(1.0, 1.0, 1e-5)
    noise = gaussian_noise(100_000, 1.0, 1.0, 1e-5, rng)
    empirical = float(np.std(noise))
    assert abs(empirical - sigma) / sigma < 0.05


def test_accountant_charges_and_tracks_remaining():
    acc = PrivacyAccountant(limit=PrivacyBudget(2.0, 1e-4))
    acc.charge(PrivacyBudget(0.5, 1e-5), purpose="q1", subject="s")
    acc.charge(PrivacyBudget(0.5, 1e-5), purpose="q2", subject="s")
    assert acc.spent.epsilon == pytest.approx(1.0)
    assert acc.remaining.epsilon == pytest.approx(1.0)


def test_accountant_rejects_over_budget():
    acc = PrivacyAccountant(limit=PrivacyBudget(1.0, 1e-5))
    acc.charge(PrivacyBudget(0.9, 1e-6), purpose="q", subject="s")
    with pytest.raises(PermissionError):
        acc.charge(PrivacyBudget(0.5, 1e-6), purpose="q2", subject="s")


def test_gaussian_rdp_cost_exact():
    # RDP(alpha) = alpha * sens^2 / (2 sigma^2): sens=1, sigma=1, alpha=2 -> 1.0
    cost = gaussian_rdp_cost(1.0, 1.0, alphas=[2.0])
    assert cost[2.0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        gaussian_rdp_cost(1.0, 0.0)


def test_laplace_rdp_cost_bounded_by_epsilon():
    # Pure epsilon-DP implies RDP(alpha) <= epsilon for every order ...
    cost = laplace_rdp_cost(8.0)
    assert all(v <= 8.0 for v in cost.values())
    # ... and RDP -> epsilon as alpha grows (no overflow at large orders).
    assert cost[64.0] == pytest.approx(8.0, rel=0.05)
    with pytest.raises(ValueError):
        laplace_rdp_cost(0.0)


def test_rdp_conversion_tighter_than_basic_sum():
    # 30 Gaussian rounds at (0.5, 1e-6): basic sum = 15, RDP says ~2.
    sigma = gaussian_sigma(1.0, 0.5, 1e-6)
    totals: dict[float, float] = {}
    for _ in range(30):
        for a, v in gaussian_rdp_cost(1.0, sigma).items():
            totals[a] = totals.get(a, 0.0) + v
    assert rdp_to_epsilon(totals, 1e-3) < 15.0
    assert rdp_to_epsilon(totals, 1e-3) == pytest.approx(2.06, rel=0.1)


def test_accountant_enforces_rdp_when_fully_tracked():
    acc = PrivacyAccountant(limit=PrivacyBudget(10.0, 1e-3))
    sigma = gaussian_sigma(1.0, 0.5, 1e-6)
    for i in range(30):
        acc.charge(
            PrivacyBudget(0.5, 1e-6),
            purpose=f"round{i}",
            subject="fl",
            rdp=gaussian_rdp_cost(1.0, sigma),
        )
    # basic sum (15) exceeds the limit, but RDP-converted spend (~2) does not
    assert acc.spent.epsilon == pytest.approx(15.0)
    assert acc.rdp_epsilon() < 10.0
    assert acc.rdp_complete is True


def test_accountant_allows_when_basic_fits_but_rdp_does_not():
    # RDP conversion overhead dominates for a few Laplace releases: basic
    # (2.0, 0.0) fits the limit even though RDP-converted (~2.2) exceeds it.
    acc = PrivacyAccountant(limit=PrivacyBudget(2.0, 1e-3))
    for i in range(2):
        acc.charge(
            PrivacyBudget(1.0, 0.0), purpose=f"q{i}", subject="s",
            rdp=laplace_rdp_cost(1.0),
        )
    assert acc.spent.epsilon == pytest.approx(2.0)
    with pytest.raises(PermissionError):  # third fits neither regime
        acc.charge(
            PrivacyBudget(1.0, 0.0), purpose="q2", subject="s",
            rdp=laplace_rdp_cost(1.0),
        )


def test_accountant_falls_back_to_basic_without_rdp():
    acc = PrivacyAccountant(limit=PrivacyBudget(1.0, 1e-5))
    acc.charge(PrivacyBudget(0.9, 1e-6), purpose="q", subject="s")
    assert acc.rdp_complete is False
    assert acc.rdp_epsilon() == 0.0
    with pytest.raises(PermissionError):
        acc.charge(PrivacyBudget(0.5, 1e-6), purpose="q2", subject="s")


def test_platform_reports_rdp_spend():
    from sentrylink.platform import SentryLinkPlatform

    p = SentryLinkPlatform()
    ids = [p.join(f"O{i}", "retail", "g-rdp").org_id for i in range(3)]
    p.histogram("g-rdp", "retail", {oid: [4, 2] for oid in ids}, epsilon=1.0)
    report = p.budget_report()
    assert report["rdp_complete"] is True
    assert report["rdp_epsilon_spent"] > 0.0
    assert np.isfinite(report["rdp_epsilon_spent"])
