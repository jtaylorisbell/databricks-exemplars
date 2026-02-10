"""Feature engineering constants and helpers for fare prediction.

These define the canonical business rules for feature engineering.
The PySpark implementation in train.py mirrors this logic.
Tests import from this module to validate the rules against a single source of truth.
"""

FEATURE_COLUMNS = [
    "trip_distance",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
]

TARGET_COLUMN = "fare_amount"

# Fare outlier thresholds
FARE_MIN = 0
FARE_MAX = 200

# Weekend: Sunday (1) and Saturday (7) in Spark's dayofweek
WEEKEND_DAYS = {1, 7}

# Rush hour ranges (inclusive)
RUSH_HOUR_MORNING = (7, 9)
RUSH_HOUR_EVENING = (16, 19)


def is_weekend(dayofweek: int) -> int:
    """Return 1 if day is weekend (Sunday=1, Saturday=7), 0 otherwise."""
    return 1 if dayofweek in WEEKEND_DAYS else 0


def is_rush_hour(hour: int) -> int:
    """Return 1 if hour falls within rush hour ranges, 0 otherwise."""
    morning_start, morning_end = RUSH_HOUR_MORNING
    evening_start, evening_end = RUSH_HOUR_EVENING
    if (morning_start <= hour <= morning_end) or (evening_start <= hour <= evening_end):
        return 1
    return 0


def is_valid_fare(fare: float) -> bool:
    """Return True if fare is within valid range."""
    return FARE_MIN < fare < FARE_MAX
