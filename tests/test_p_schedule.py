import pytest

from utils.reward_utils import get_privacy_penalty


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, 20),
        (1, 20),
        (349, 20),
        (350, 25),
        (699, 25),
        (700, 30),
        (1049, 30),
        (1050, 35),
        (1399, 35),
        (1400, 40),
        (1750, 40),
        (1_000_000, 40),
    ],
)
def test_privacy_penalty_schedule(step, expected):
    assert get_privacy_penalty(step) == expected


def test_privacy_penalty_rejects_negative_steps():
    with pytest.raises(ValueError):
        get_privacy_penalty(-1)


@pytest.mark.parametrize("step", [1.5, "350", True])
def test_privacy_penalty_rejects_non_integer_steps(step):
    with pytest.raises(TypeError):
        get_privacy_penalty(step)


def test_privacy_penalty_supports_an_explicit_custom_schedule():
    assert get_privacy_penalty(0, initial=2, interval=10, increment=3, maximum=8) == 2
    assert get_privacy_penalty(10, initial=2, interval=10, increment=3, maximum=8) == 5
    assert get_privacy_penalty(20, initial=2, interval=10, increment=3, maximum=8) == 8
    assert get_privacy_penalty(30, initial=2, interval=10, increment=3, maximum=8) == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial": -1},
        {"interval": 0},
        {"increment": -1},
        {"initial": 20, "maximum": 19},
        {"maximum": 40.0},
    ],
)
def test_privacy_penalty_rejects_invalid_schedule_parameters(kwargs):
    with pytest.raises((TypeError, ValueError)):
        get_privacy_penalty(0, **kwargs)
