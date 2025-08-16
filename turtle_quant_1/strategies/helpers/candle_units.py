"""Helper functions for working with candle unit conversions."""

from turtle_quant_1.config import CANDLE_UNIT

_SUPPORTED_CANDLE_UNITS = ["HOUR", "DAY", "WEEK", "MONTH"]
assert CANDLE_UNIT in _SUPPORTED_CANDLE_UNITS


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

    # NOTE: The atomic unit has been arbitrarily chosen to be 1 hour.
    atomic_units = 0
    if from_unit == "HOUR":
        atomic_units = units
    if from_unit == "DAY":
        # NOTE: Assume there are 6 hours in a working day.
        atomic_units = units * 6
    if from_unit == "WEEK":
        # NOTE: Assume there are 5 working days in a week.
        atomic_units = units * 6 * 5
    if from_unit == "MONTH":
        # NOTE: Assume there are 20 working days in a month.
        atomic_units = units * 6 * 20

    if to_unit == "HOUR":
        return atomic_units
    if to_unit == "DAY":
        return round(atomic_units / 6)
    if to_unit == "WEEK":
        return round(atomic_units / (6 * 5))
    if to_unit == "MONTH":
        return round(atomic_units / (6 * 20))

    raise ValueError(f"Invalid unit conversion: {from_unit} to {to_unit}")


def hours_to_units(hours: int) -> int:
    """Convert hours to units.

    Args:
        hours: The number of hours to convert.

    Returns:
        The number of units in the new unit.
    """
    return convert_units(hours, "HOUR", CANDLE_UNIT)


def days_to_units(days: int) -> int:
    """Convert days to units.

    Args:
        days: The number of days to convert.

    Returns:
        The number of units in the new unit.
    """
    return convert_units(days, "DAY", CANDLE_UNIT)


def weeks_to_units(weeks: int) -> int:
    """Convert weeks to units.

    Args:
        weeks: The number of weeks to convert.

    Returns:
        The number of units in the new unit.
    """
    return convert_units(weeks, "WEEK", CANDLE_UNIT)


def months_to_units(months: int) -> int:
    """Convert months to units.

    Args:
        months: The number of months to convert.

    Returns:
        The number of units in the new unit.
    """
    return convert_units(months, "MONTH", CANDLE_UNIT)
