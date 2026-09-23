import numpy as np
import pytest

from sentrylink.crypto.differential_privacy import (
    PrivacyAccountant,
    PrivacyBudget,
    gaussian_noise,
    gaussian_sigma,
    laplace_noise,
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
