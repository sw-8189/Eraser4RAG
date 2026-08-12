import math

import pytest

from utils.reward_utils import (
    compute_reward,
    compute_retention_rates,
    privacy_utility_reward,
    retention_rate,
    reward_from_triples,
)


PUBLIC = [["h1", "r1", "t1"], ["h2", "r2", "t2"]]
PRIVATE = [["secret", "relation", "value"]]


def test_retention_rate_uses_unique_triple_sets():
    predicted = [PUBLIC[0], PUBLIC[0], PRIVATE[0]]
    reference = [PUBLIC[0], PUBLIC[0], PUBLIC[1]]
    assert retention_rate(predicted, reference) == 0.5


def test_empty_reference_value_is_explicit():
    assert retention_rate([], [], empty_reference_value=0.0) == 0.0
    assert retention_rate([], [], empty_reference_value=1.0) == 1.0


def test_compute_retention_rates_applies_public_private_empty_conventions():
    rates = compute_retention_rates([], [], [])
    assert rates == (1.0, 0.0)
    assert rates.public == 1.0
    assert rates.private == 0.0


def test_compute_retention_rates_matches_reference_sets():
    rates = compute_retention_rates([PUBLIC[0], PRIVATE[0]], PUBLIC, PRIVATE)
    assert rates.public == 0.5
    assert rates.private == 1.0


def test_reward_boundaries():
    assert compute_reward(1.0, 0.0, 20) == 1.0
    assert privacy_utility_reward(1.0, 0.0, 20) == 1.0
    assert privacy_utility_reward(0.0, 0.0, 20) == 0.0
    assert privacy_utility_reward(0.0, 1.0, 20) == 0.0


def test_larger_penalty_reduces_reward_when_private_information_remains():
    lower_penalty = privacy_utility_reward(0.8, 0.2, 20)
    higher_penalty = privacy_utility_reward(0.8, 0.2, 40)
    assert higher_penalty < lower_penalty
    assert lower_penalty == pytest.approx(0.8 * math.exp(-4.0))


def test_reward_from_triples_uses_paper_formula():
    predicted = [PUBLIC[0], PRIVATE[0]]
    expected = 0.5 * math.exp(-20.0)
    assert reward_from_triples(predicted, PUBLIC, PRIVATE, 20) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("r_pub", "r_pri", "penalty"),
    [
        (-0.1, 0.0, 20),
        (1.1, 0.0, 20),
        (1.0, -0.1, 20),
        (1.0, 1.1, 20),
        (1.0, 0.0, -1),
        (float("nan"), 0.0, 20),
    ],
)
def test_reward_rejects_invalid_values(r_pub, r_pri, penalty):
    with pytest.raises((TypeError, ValueError)):
        privacy_utility_reward(r_pub, r_pri, penalty)
