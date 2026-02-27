"""Helper functions for working with candle unit conversions."""

from turtle_quant_1.config import CANDLE_UNIT

_SUPPORTED_CANDLE_UNITS = ["5M", "15M", "30M", "1H", "2H", "4H", "1D", "1W"]
assert CANDLE_UNIT in _SUPPORTED_CANDLE_UNITS

# Conversion factors relative to 5M (the atomic unit).
# NOTE: Based on assumption(s): 6 working hours per day, 5 working days per week.
_UNIT_TO_5M = {
    "5M": 1,
    "15M": 3,
    "30M": 6,
    "1H": 12,
    "2H": 24,
    "4H": 48,
    "1D": 72,  # 6-hour day *12
    "1W": 360,  # 5-day week * 12
}


def convert_units(units: int, from_unit: str, to_unit: str) -> int:
    """Approximately convert units between different candle units.

    Args:
        units: The number of units to convert.
        from_unit: The unit to convert from.
        to_unit: The unit to convert to.

    Returns:
        The number of units in the new unit.
    """
    assert from_unit in _SUPPORTED_CANDLE_UNITS
    assert to_unit in _SUPPORTED_CANDLE_UNITS

    if from_unit == to_unit:
        return units

    atomic_units = units * _UNIT_TO_5M[from_unit]
    return round(atomic_units / _UNIT_TO_5M[to_unit])
