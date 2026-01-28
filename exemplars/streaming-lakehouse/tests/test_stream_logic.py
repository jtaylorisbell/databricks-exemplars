"""Unit tests for streaming transformation logic.

Run with: uv run pytest tests/
"""

import pytest
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("streaming-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


class TestStreamFiltering:
    """Tests for stream filtering logic."""

    def test_null_event_id_filtered(self, spark):
        """Events with null event_id should be filtered out."""
        data = [
            ("evt1", "click", "user1"),
            (None, "view", "user2"),
            ("evt3", "purchase", None),
        ]
        df = spark.createDataFrame(data, ["event_id", "event_type", "user_id"])

        result = df.filter(F.col("event_id").isNotNull())

        assert result.count() == 2

    def test_null_event_type_filtered(self, spark):
        """Events with null event_type should be filtered out."""
        data = [
            ("evt1", "click", "user1"),
            ("evt2", None, "user2"),
            ("evt3", "purchase", "user3"),
        ]
        df = spark.createDataFrame(data, ["event_id", "event_type", "user_id"])

        result = df.filter(F.col("event_type").isNotNull())

        assert result.count() == 2


class TestTimeExtraction:
    """Tests for time-based column extraction."""

    def test_event_date_extraction(self, spark):
        """Event date should be extracted from timestamp."""
        data = [(datetime(2024, 1, 15, 14, 30, 0),)]
        df = spark.createDataFrame(data, ["event_time"])

        result = df.withColumn("event_date", F.to_date("event_time"))
        row = result.first()

        assert str(row.event_date) == "2024-01-15"

    def test_event_hour_extraction(self, spark):
        """Event hour should be extracted from timestamp."""
        data = [(datetime(2024, 1, 15, 14, 30, 0),)]
        df = spark.createDataFrame(data, ["event_time"])

        result = df.withColumn("event_hour", F.hour("event_time"))
        row = result.first()

        assert row.event_hour == 14


class TestAggregations:
    """Tests for windowed aggregation logic."""

    def test_event_count_aggregation(self, spark):
        """Event counts should be aggregated by type."""
        data = [
            ("evt1", "click", 10.0),
            ("evt2", "click", 20.0),
            ("evt3", "view", 5.0),
        ]
        df = spark.createDataFrame(data, ["event_id", "event_type", "value"])

        result = (
            df.groupBy("event_type")
            .agg(
                F.count("event_id").alias("event_count"),
                F.sum("value").alias("total_value"),
            )
            .orderBy("event_type")
        )

        rows = result.collect()
        assert rows[0].event_count == 2  # click
        assert rows[0].total_value == 30.0
        assert rows[1].event_count == 1  # view

    def test_average_value_aggregation(self, spark):
        """Average values should be calculated correctly."""
        data = [
            ("evt1", "click", 10.0),
            ("evt2", "click", 20.0),
            ("evt3", "click", 30.0),
        ]
        df = spark.createDataFrame(data, ["event_id", "event_type", "value"])

        result = df.groupBy("event_type").agg(F.avg("value").alias("avg_value"))
        row = result.first()

        assert row.avg_value == 20.0
