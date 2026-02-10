"""Unit tests for fare prediction feature engineering.

Tests import business rules from src/features.py — the single source of truth
for feature constants and transformation logic. This ensures tests stay in sync
with the rules used by the PySpark implementation in train.py.

Run with: uv run pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    FARE_MIN,
    FARE_MAX,
    WEEKEND_DAYS,
    is_weekend,
    is_rush_hour,
    is_valid_fare,
)


class TestFeatureTransformations:
    """Tests for feature transformation functions."""

    def test_weekend_flag(self):
        """Weekend flag should be 1 for Saturday (7) and Sunday (1)."""
        dayofweek_values = [1, 2, 3, 4, 5, 6, 7]  # Sun=1, Sat=7
        expected = [1, 0, 0, 0, 0, 0, 1]

        results = [is_weekend(d) for d in dayofweek_values]

        assert results == expected

    def test_rush_hour_flag(self):
        """Rush hour should be flagged for 7-9 AM and 4-7 PM."""
        hours = [6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20]
        expected = [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]

        results = [is_rush_hour(h) for h in hours]

        assert results == expected

    def test_fare_outlier_filtering(self):
        """Fares outside valid range should be filtered."""
        fares = [5.0, 50.0, 150.0, 250.0, -5.0, 0.0]

        valid_fares = [f for f in fares if is_valid_fare(f)]

        assert valid_fares == [5.0, 50.0, 150.0]

    def test_fare_thresholds(self):
        """Fare thresholds should match expected bounds."""
        assert FARE_MIN == 0
        assert FARE_MAX == 200

    def test_weekend_days(self):
        """Weekend days should be Sunday (1) and Saturday (7)."""
        assert WEEKEND_DAYS == {1, 7}


class TestFeatureValidation:
    """Tests for feature data validation."""

    def test_feature_columns_complete(self):
        """All expected feature columns should be defined."""
        expected = {
            "trip_distance", "pickup_hour", "pickup_dayofweek",
            "pickup_month", "is_weekend", "is_rush_hour",
        }
        assert set(FEATURE_COLUMNS) == expected

    def test_feature_dtypes(self):
        """Features should have expected numeric types."""
        df = pd.DataFrame({
            "trip_distance": [1.5, 2.0, 3.5],
            "pickup_hour": [8, 12, 18],
            "pickup_dayofweek": [1, 3, 5],
            "pickup_month": [1, 6, 12],
            "is_weekend": [0, 0, 1],
            "is_rush_hour": [1, 0, 1],
        })

        assert df["trip_distance"].dtype == np.float64
        assert df["pickup_hour"].dtype == np.int64
        assert df["is_weekend"].dtype == np.int64

    def test_no_null_features(self):
        """Features should not contain nulls after transformation."""
        df = pd.DataFrame({
            "trip_distance": [1.5, 2.0, None],
            "pickup_hour": [8, 12, 18],
        })

        df_clean = df.dropna()

        assert df_clean["trip_distance"].isna().sum() == 0
        assert len(df_clean) == 2


class TestModelPredictions:
    """Tests for model prediction logic."""

    def test_model_trains_without_error(self):
        """Model should train without errors on sample data."""
        X = pd.DataFrame({col: [1.0, 2.0, 5.0, 10.0, 15.0] for col in FEATURE_COLUMNS})
        X["trip_distance"] = [1.0, 2.0, 5.0, 10.0, 15.0]
        y = [10.0, 15.0, 25.0, 40.0, 55.0]

        model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_predictions_are_positive(self):
        """Fare predictions should always be positive."""
        X = pd.DataFrame({
            "trip_distance": [1.0, 5.0, 10.0],
            "pickup_hour": [8, 12, 17],
            "pickup_dayofweek": [2, 3, 4],
            "pickup_month": [1, 6, 12],
            "is_weekend": [0, 0, 0],
            "is_rush_hour": [1, 0, 1],
        })
        y = [10.0, 25.0, 45.0]

        model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        assert all(p > 0 for p in predictions)

    def test_longer_trips_cost_more(self):
        """Longer trips should generally have higher predicted fares."""
        X_train = pd.DataFrame({
            "trip_distance": [1.0, 5.0, 10.0, 15.0, 20.0],
            "pickup_hour": [12, 12, 12, 12, 12],
            "pickup_dayofweek": [3, 3, 3, 3, 3],
            "pickup_month": [6, 6, 6, 6, 6],
            "is_weekend": [0, 0, 0, 0, 0],
            "is_rush_hour": [0, 0, 0, 0, 0],
        })
        y_train = [8.0, 18.0, 32.0, 48.0, 65.0]

        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Predict for short and long trips
        X_test = pd.DataFrame({
            "trip_distance": [2.0, 12.0],
            "pickup_hour": [12, 12],
            "pickup_dayofweek": [3, 3],
            "pickup_month": [6, 6],
            "is_weekend": [0, 0],
            "is_rush_hour": [0, 0],
        })

        predictions = model.predict(X_test)
        assert predictions[1] > predictions[0]  # Longer trip costs more
