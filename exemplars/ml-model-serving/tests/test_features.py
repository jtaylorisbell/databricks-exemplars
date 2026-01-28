"""Unit tests for feature engineering logic.

Run with: uv run pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class TestFeatureTransformations:
    """Tests for feature transformation functions."""

    def test_ratio_calculation(self):
        """Feature ratio should be calculated correctly."""
        df = pd.DataFrame({
            "value_a": [10, 20, 30],
            "value_b": [2, 4, 5],
        })

        df["feature_ratio"] = df["value_a"] / df["value_b"]

        assert df["feature_ratio"].tolist() == [5.0, 5.0, 6.0]

    def test_log_transform(self):
        """Log transform should handle zeros correctly."""
        df = pd.DataFrame({"amount": [0, 1, 100]})

        # log1p handles zeros
        df["feature_log"] = np.log1p(df["amount"])

        assert df["feature_log"][0] == 0.0  # log1p(0) = 0
        assert df["feature_log"][1] == pytest.approx(0.693, rel=0.01)

    def test_binning_logic(self):
        """Binning should categorize correctly."""
        amounts = [50, 500, 5000]

        def bin_amount(x):
            if x < 100:
                return "low"
            elif x < 1000:
                return "medium"
            else:
                return "high"

        results = [bin_amount(a) for a in amounts]

        assert results == ["low", "medium", "high"]

    def test_label_encoding(self):
        """Categorical encoding should be consistent."""
        categories = ["low", "medium", "high", "low", "medium"]

        le = LabelEncoder()
        encoded = le.fit_transform(categories)

        # Same categories should have same encoding
        assert encoded[0] == encoded[3]  # both "low"
        assert encoded[1] == encoded[4]  # both "medium"


class TestFeatureValidation:
    """Tests for feature data validation."""

    def test_no_null_features(self):
        """Features should not contain nulls after transformation."""
        df = pd.DataFrame({
            "value_a": [10, 20, None],
            "value_b": [2, 4, 5],
        })

        # This would fail if nulls propagate
        df_clean = df.dropna()
        df_clean["feature_ratio"] = df_clean["value_a"] / df_clean["value_b"]

        assert df_clean["feature_ratio"].isna().sum() == 0

    def test_feature_dtypes(self):
        """Features should have expected data types."""
        df = pd.DataFrame({
            "feature_ratio": [1.5, 2.0, 3.5],
            "feature_log": [0.0, 0.5, 1.0],
            "feature_binned": ["low", "medium", "high"],
        })

        assert df["feature_ratio"].dtype == np.float64
        assert df["feature_log"].dtype == np.float64
        assert df["feature_binned"].dtype == object  # string
