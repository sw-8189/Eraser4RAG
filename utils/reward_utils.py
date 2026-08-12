"""Pure reward helpers shared by PPO code and CPU unit tests."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import NamedTuple

from utils.triple_utils import normalize_triple_collection


INITIAL_PRIVACY_PENALTY = 20
PRIVACY_PENALTY_INCREMENT = 5
PRIVACY_PENALTY_INTERVAL = 350
MAX_PRIVACY_PENALTY = 40


class RetentionRates(NamedTuple):
    r_pub: float
    r_pri: float

    @property
    def public(self) -> float:
        """Backward-compatible descriptive alias for ``r_pub``."""

        return self.r_pub

    @property
    def private(self) -> float:
        """Backward-compatible descriptive alias for ``r_pri``."""

        return self.r_pri


def _validate_unit_interval(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return numeric


def retention_rate(
    predicted_triples: object,
    reference_triples: object,
    *,
    empty_reference_value: float = 0.0,
) -> float:
    """Return the fraction of unique reference triples found in predictions."""

    empty_value = _validate_unit_interval(empty_reference_value, "empty_reference_value")
    predicted = {tuple(triple) for triple in normalize_triple_collection(predicted_triples)}
    reference = {tuple(triple) for triple in normalize_triple_collection(reference_triples)}
    if not reference:
        return empty_value
    return len(predicted & reference) / len(reference)


def compute_retention_rates(
    predicted_triples: object,
    public_reference: object,
    private_reference: object,
) -> RetentionRates:
    """Return ``(r_pub, r_pri)`` for the paper's reference triple sets.

    By definition, an empty public reference yields ``r_pub=1`` and an empty
    private reference yields ``r_pri=0``.  The named tuple can be unpacked as a
    regular two-tuple.
    """

    return RetentionRates(
        r_pub=retention_rate(
            predicted_triples, public_reference, empty_reference_value=1.0
        ),
        r_pri=retention_rate(
            predicted_triples, private_reference, empty_reference_value=0.0
        ),
    )


def privacy_utility_reward(r_pub: Real, r_pri: Real, penalty: Real) -> float:
    """Compute ``r_pub * exp(-penalty * r_pri)`` exactly as in the paper."""

    public = _validate_unit_interval(r_pub, "r_pub")
    private = _validate_unit_interval(r_pri, "r_pri")
    if isinstance(penalty, bool) or not isinstance(penalty, Real):
        raise TypeError("penalty must be a real number")
    numeric_penalty = float(penalty)
    if not math.isfinite(numeric_penalty) or numeric_penalty < 0:
        raise ValueError("penalty must be finite and non-negative")
    return public * math.exp(-numeric_penalty * private)


def compute_reward(r_pub: Real, r_pri: Real, p: Real) -> float:
    """Stable API for ``r_pub * exp(-p * r_pri)``."""

    return privacy_utility_reward(r_pub, r_pri, p)


def reward_from_triples(
    predicted_triples: object,
    public_reference: object,
    private_reference: object,
    penalty: Real,
) -> float:
    rates = compute_retention_rates(
        predicted_triples, public_reference, private_reference
    )
    return compute_reward(rates.r_pub, rates.r_pri, penalty)


def get_privacy_penalty(
    step: int,
    initial: int = INITIAL_PRIVACY_PENALTY,
    interval: int = PRIVACY_PENALTY_INTERVAL,
    increment: int = PRIVACY_PENALTY_INCREMENT,
    maximum: int = MAX_PRIVACY_PENALTY,
) -> int:
    """Return a stepped privacy penalty capped permanently at ``maximum``.

    Defaults implement the v2 schedule: boundaries at 350, 700, 1050, and
    1400 outer PPO steps, starting at 20 and capped at 40.  Step zero is
    accepted and uses ``initial``.
    """

    if isinstance(step, bool) or not isinstance(step, Integral):
        raise TypeError("step must be an integer")
    if step < 0:
        raise ValueError("step must be non-negative")
    parameters = {
        "initial": initial,
        "interval": interval,
        "increment": increment,
        "maximum": maximum,
    }
    for name, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
    if initial < 0:
        raise ValueError("initial must be non-negative")
    if interval <= 0:
        raise ValueError("interval must be positive")
    if increment < 0:
        raise ValueError("increment must be non-negative")
    if maximum < initial:
        raise ValueError("maximum must be greater than or equal to initial")

    completed_intervals = int(step) // int(interval)
    return min(int(initial) + completed_intervals * int(increment), int(maximum))


__all__ = [
    "INITIAL_PRIVACY_PENALTY",
    "MAX_PRIVACY_PENALTY",
    "PRIVACY_PENALTY_INCREMENT",
    "PRIVACY_PENALTY_INTERVAL",
    "RetentionRates",
    "compute_reward",
    "compute_retention_rates",
    "get_privacy_penalty",
    "privacy_utility_reward",
    "retention_rate",
    "reward_from_triples",
]
