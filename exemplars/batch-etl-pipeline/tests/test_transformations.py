"""Unit tests for transformation logic.

These tests validate the transformation functions used in the pipeline.
The pipeline code itself runs only in Databricks, but we can test the
transformation logic locally using PySpark.

Run with: uv run pytest tests/ -v
"""

import pytest
from datetime import datetime, date
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    TimestampType,
)


@pytest.fixture(scope="session")
def spark():
    """Create a SparkSession for testing."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("batch-etl-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


class TestSilverTransactionTransformations:
    """Tests for silver_transactions transformation logic."""

    def test_date_extraction(self, spark):
        """Verify date components are correctly extracted from timestamp."""
        data = [(datetime(2024, 5, 15, 14, 30, 0),)]
        df = spark.createDataFrame(data, ["dateTime"])

        result = (
            df
            .withColumn("transaction_date", F.to_date("dateTime"))
            .withColumn("transaction_hour", F.hour("dateTime"))
            .withColumn("day_of_week", F.dayofweek("dateTime"))
        )

        row = result.first()
        assert str(row.transaction_date) == "2024-05-15"
        assert row.transaction_hour == 14
        assert row.day_of_week == 4  # Wednesday

    def test_weekend_detection(self, spark):
        """Verify weekend flag is set correctly."""
        data = [
            (datetime(2024, 5, 11, 10, 0, 0),),  # Saturday
            (datetime(2024, 5, 12, 10, 0, 0),),  # Sunday
            (datetime(2024, 5, 13, 10, 0, 0),),  # Monday
        ]
        df = spark.createDataFrame(data, ["dateTime"])

        result = (
            df
            .withColumn("day_of_week", F.dayofweek("dateTime"))
            .withColumn(
                "is_weekend",
                F.when(F.col("day_of_week").isin(1, 7), True).otherwise(False)
            )
        )

        rows = result.collect()
        assert rows[0].is_weekend is True   # Saturday (day 7)
        assert rows[1].is_weekend is True   # Sunday (day 1)
        assert rows[2].is_weekend is False  # Monday (day 2)

    def test_payment_method_standardization(self, spark):
        """Verify payment method is uppercased and trimmed."""
        data = [
            ("  visa  ",),
            ("MasterCard",),
            ("AMEX",),
        ]
        df = spark.createDataFrame(data, ["paymentMethod"])

        result = df.withColumn(
            "payment_method_clean",
            F.upper(F.trim(F.col("paymentMethod")))
        )

        rows = result.collect()
        assert rows[0].payment_method_clean == "VISA"
        assert rows[1].payment_method_clean == "MASTERCARD"
        assert rows[2].payment_method_clean == "AMEX"

    def test_card_number_masking(self, spark):
        """Verify card numbers are masked to show only last 4 digits."""
        data = [(378154478982993,), (2244626981238094,)]
        df = spark.createDataFrame(data, ["cardNumber"])

        result = df.withColumn(
            "card_number_masked",
            F.concat(
                F.lit("****-****-****-"),
                F.substring(F.col("cardNumber").cast("string"), -4, 4)
            )
        )

        rows = result.collect()
        assert rows[0].card_number_masked == "****-****-****-2993"
        assert rows[1].card_number_masked == "****-****-****-8094"

    def test_data_quality_rules(self, spark):
        """Verify data quality filtering logic."""
        data = [
            (1, 100, 200, 5, 50),   # Valid
            (None, 100, 200, 5, 50),  # Invalid: null transactionID
            (2, None, 200, 5, 50),   # Invalid: null customerID
            (3, 100, None, 5, 50),   # Invalid: null franchiseID
            (4, 100, 200, 0, 50),    # Invalid: zero quantity
            (5, 100, 200, 5, -10),   # Invalid: negative totalPrice
        ]
        df = spark.createDataFrame(
            data,
            ["transactionID", "customerID", "franchiseID", "quantity", "totalPrice"]
        )

        # Apply quality rules (simulating @dp.expect_all_or_drop)
        result = (
            df
            .filter(F.col("transactionID").isNotNull())
            .filter(F.col("customerID").isNotNull())
            .filter(F.col("franchiseID").isNotNull())
            .filter(F.col("quantity") > 0)
            .filter(F.col("totalPrice") > 0)
        )

        assert result.count() == 1
        assert result.first().transactionID == 1


class TestSilverCustomerTransformations:
    """Tests for silver_customers transformation logic."""

    def test_full_name_creation(self, spark):
        """Verify full name is correctly concatenated."""
        data = [("John", "Doe"), ("Jane", "Smith")]
        df = spark.createDataFrame(data, ["first_name", "last_name"])

        result = df.withColumn(
            "full_name",
            F.concat_ws(" ", F.col("first_name"), F.col("last_name"))
        )

        rows = result.collect()
        assert rows[0].full_name == "John Doe"
        assert rows[1].full_name == "Jane Smith"

    def test_region_mapping(self, spark):
        """Verify continent to region code mapping."""
        data = [
            ("North America",),
            ("Europe",),
            ("Asia",),
            ("South America",),
            ("Africa",),
        ]
        df = spark.createDataFrame(data, ["continent"])

        result = df.withColumn(
            "region",
            F.when(F.col("continent") == "North America", "NA")
            .when(F.col("continent") == "Europe", "EU")
            .when(F.col("continent") == "Asia", "APAC")
            .otherwise("OTHER")
        )

        rows = result.collect()
        assert rows[0].region == "NA"
        assert rows[1].region == "EU"
        assert rows[2].region == "APAC"
        assert rows[3].region == "OTHER"
        assert rows[4].region == "OTHER"

    def test_email_validation(self, spark):
        """Verify email validation logic."""
        data = [
            ("user@example.com",),
            ("invalid-email",),
            (None,),
            ("another@test.org",),
        ]
        df = spark.createDataFrame(data, ["email_address"])

        result = df.filter(
            F.col("email_address").isNotNull() &
            F.col("email_address").like("%@%")
        )

        assert result.count() == 2


class TestSilverFranchiseTransformations:
    """Tests for silver_franchises transformation logic."""

    def test_size_category_mapping(self, spark):
        """Verify franchise size categorization."""
        data = [("Small",), ("Medium",), ("Large",), ("Extra Large",)]
        df = spark.createDataFrame(data, ["size"])

        result = df.withColumn(
            "size_category",
            F.when(F.col("size") == "Small", "S")
            .when(F.col("size") == "Medium", "M")
            .when(F.col("size") == "Large", "L")
            .otherwise("Unknown")
        )

        rows = result.collect()
        assert rows[0].size_category == "S"
        assert rows[1].size_category == "M"
        assert rows[2].size_category == "L"
        assert rows[3].size_category == "Unknown"


class TestGoldAggregations:
    """Tests for gold layer aggregation logic."""

    def test_daily_sales_summary(self, spark):
        """Verify daily sales metrics are calculated correctly."""
        data = [
            (1, 100, 200, date(2024, 5, 1), 50, 3),
            (2, 101, 200, date(2024, 5, 1), 75, 2),
            (3, 100, 200, date(2024, 5, 2), 100, 4),
        ]
        df = spark.createDataFrame(
            data,
            ["transactionID", "customerID", "franchiseID", "transaction_date", "totalPrice", "quantity"]
        )

        result = (
            df.groupBy("transaction_date", "franchiseID")
            .agg(
                F.sum("totalPrice").alias("total_revenue"),
                F.count("transactionID").alias("transaction_count"),
                F.avg("totalPrice").alias("avg_transaction_value"),
                F.countDistinct("customerID").alias("unique_customers"),
                F.sum("quantity").alias("total_items_sold"),
            )
            .orderBy("transaction_date")
        )

        rows = result.collect()
        # May 1: 2 transactions, revenue 125, 2 unique customers
        assert rows[0].total_revenue == 125
        assert rows[0].transaction_count == 2
        assert rows[0].unique_customers == 2
        assert rows[0].total_items_sold == 5
        # May 2: 1 transaction, revenue 100
        assert rows[1].total_revenue == 100
        assert rows[1].transaction_count == 1

    def test_customer_lifetime_value(self, spark):
        """Verify customer LTV calculation."""
        data = [
            (100, 50, date(2024, 5, 1)),
            (100, 75, date(2024, 5, 5)),
            (100, 100, date(2024, 5, 10)),
            (101, 200, date(2024, 5, 3)),
        ]
        df = spark.createDataFrame(
            data,
            ["customerID", "totalPrice", "transaction_date"]
        )

        result = (
            df.groupBy("customerID")
            .agg(
                F.sum("totalPrice").alias("lifetime_value"),
                F.count("*").alias("total_transactions"),
                F.min("transaction_date").alias("first_purchase_date"),
                F.max("transaction_date").alias("last_purchase_date"),
            )
            .orderBy("customerID")
        )

        rows = result.collect()
        # Customer 100: 3 transactions, $225 LTV
        assert rows[0].lifetime_value == 225
        assert rows[0].total_transactions == 3
        # Customer 101: 1 transaction, $200 LTV
        assert rows[1].lifetime_value == 200
        assert rows[1].total_transactions == 1

    def test_product_performance(self, spark):
        """Verify product-level aggregations."""
        data = [
            ("Golden Gate Ginger", 5, 3, 15, 200),
            ("Golden Gate Ginger", 10, 3, 30, 201),
            ("Austin Almond Biscotti", 8, 3, 24, 200),
        ]
        df = spark.createDataFrame(
            data,
            ["product", "quantity", "unitPrice", "totalPrice", "franchiseID"]
        )

        result = (
            df.groupBy("product")
            .agg(
                F.sum("quantity").alias("total_units_sold"),
                F.sum("totalPrice").alias("total_revenue"),
                F.count("*").alias("transaction_count"),
                F.countDistinct("franchiseID").alias("franchise_count"),
            )
            .orderBy("product")
        )

        rows = result.collect()
        # Austin Almond Biscotti
        assert rows[0].total_units_sold == 8
        assert rows[0].total_revenue == 24
        assert rows[0].franchise_count == 1
        # Golden Gate Ginger
        assert rows[1].total_units_sold == 15
        assert rows[1].total_revenue == 45
        assert rows[1].franchise_count == 2
