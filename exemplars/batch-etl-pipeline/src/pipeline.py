"""
Batch ETL Pipeline using Lakeflow Declarative Pipelines (formerly DLT).

This pipeline demonstrates the medallion architecture (bronze → silver → gold)
using the samples.bakehouse dataset. It transforms bakery sales data into
analytics-ready tables.

Data Flow:
    samples.bakehouse.* (source)
        → bronze_* (raw with audit)
        → silver_* (cleaned/enriched)
        → gold_* (aggregates)

Usage:
    Deploy via Databricks Asset Bundles:
        databricks bundle deploy
        databricks bundle run batch_etl_pipeline
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
    DoubleType,
    TimestampType,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Source catalog and schema (read-only sample data)
SOURCE_CATALOG = "samples"
SOURCE_SCHEMA = "bakehouse"

# Data quality rules
TRANSACTION_RULES = {
    "valid_transaction_id": "transactionID IS NOT NULL",
    "valid_customer_id": "customerID IS NOT NULL",
    "valid_franchise_id": "franchiseID IS NOT NULL",
    "positive_quantity": "quantity > 0",
    "positive_total": "totalPrice > 0",
}

CUSTOMER_RULES = {
    "valid_customer_id": "customerID IS NOT NULL",
    "valid_email": "email_address IS NOT NULL AND email_address LIKE '%@%'",
}


# =============================================================================
# BRONZE LAYER
# =============================================================================
# Raw data from source with minimal transformation. Adds audit metadata.
# Design choice: Materialized views since source tables are already Delta.
# =============================================================================


@dp.materialized_view(
    name="bronze_transactions",
    comment="Raw sales transactions from bakehouse sample data with audit metadata",
    table_properties={"quality": "bronze"},
)
def bronze_transactions():
    """Ingest raw transaction data with processing metadata."""
    return (
        spark.read.table(f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.sales_transactions")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_table", F.lit("samples.bakehouse.sales_transactions"))
    )


@dp.materialized_view(
    name="bronze_customers",
    comment="Raw customer data from bakehouse sample data with audit metadata",
    table_properties={"quality": "bronze"},
)
def bronze_customers():
    """Ingest raw customer data with processing metadata."""
    return (
        spark.read.table(f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.sales_customers")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_table", F.lit("samples.bakehouse.sales_customers"))
    )


@dp.materialized_view(
    name="bronze_franchises",
    comment="Raw franchise location data from bakehouse sample data with audit metadata",
    table_properties={"quality": "bronze"},
)
def bronze_franchises():
    """Ingest raw franchise data with processing metadata."""
    return (
        spark.read.table(f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.sales_franchises")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_table", F.lit("samples.bakehouse.sales_franchises"))
    )


@dp.materialized_view(
    name="bronze_suppliers",
    comment="Raw supplier data from bakehouse sample data with audit metadata",
    table_properties={"quality": "bronze"},
)
def bronze_suppliers():
    """Ingest raw supplier data with processing metadata."""
    return (
        spark.read.table(f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.sales_suppliers")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_table", F.lit("samples.bakehouse.sales_suppliers"))
    )


# =============================================================================
# SILVER LAYER
# =============================================================================
# Cleaned, validated, and enriched data. Data quality expectations enforced.
# Design choice: Drop invalid records rather than fail pipeline for resilience.
# =============================================================================


@dp.materialized_view(
    name="silver_transactions",
    comment="Cleaned transactions with validated fields and derived date columns",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",
    },
    cluster_by=["transaction_date"],
)
@dp.expect_all_or_drop(TRANSACTION_RULES)
def silver_transactions():
    """
    Clean and enrich transaction data.

    Transformations:
    - Extract date components for partitioning and analysis
    - Standardize payment method naming
    - Mask card numbers for PII protection
    - Drop records failing data quality rules
    """
    return (
        spark.read.table("LIVE.bronze_transactions")
        # Extract date components
        .withColumn("transaction_date", F.to_date("dateTime"))
        .withColumn("transaction_hour", F.hour("dateTime"))
        .withColumn("day_of_week", F.dayofweek("dateTime"))
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1, 7), True).otherwise(False))
        # Standardize payment method
        .withColumn(
            "payment_method_clean",
            F.upper(F.trim(F.col("paymentMethod")))
        )
        # Mask card number (keep last 4 digits only)
        .withColumn(
            "card_number_masked",
            F.concat(F.lit("****-****-****-"), F.substring(F.col("cardNumber").cast("string"), -4, 4))
        )
        # Drop raw card number for PII protection
        .drop("cardNumber")
        # Rename for consistency
        .withColumnRenamed("dateTime", "transaction_timestamp")
        .withColumnRenamed("paymentMethod", "payment_method_raw")
    )


@dp.materialized_view(
    name="silver_customers",
    comment="Cleaned customer data with standardized fields",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",
    },
)
@dp.expect_all_or_drop(CUSTOMER_RULES)
def silver_customers():
    """
    Clean and standardize customer data.

    Transformations:
    - Create full name field
    - Standardize country/state naming
    - Create customer segment based on location
    """
    return (
        spark.read.table("LIVE.bronze_customers")
        # Create derived fields
        .withColumn(
            "full_name",
            F.concat_ws(" ", F.col("first_name"), F.col("last_name"))
        )
        # Standardize location fields
        .withColumn("country_clean", F.upper(F.trim(F.col("country"))))
        .withColumn("state_clean", F.upper(F.trim(F.col("state"))))
        # Create regional segment
        .withColumn(
            "region",
            F.when(F.col("continent") == "North America", "NA")
            .when(F.col("continent") == "Europe", "EU")
            .when(F.col("continent") == "Asia", "APAC")
            .otherwise("OTHER")
        )
    )


@dp.materialized_view(
    name="silver_franchises",
    comment="Cleaned franchise data with supplier relationship",
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",
    },
)
@dp.expect("valid_franchise_id", "franchiseID IS NOT NULL")
@dp.expect("valid_location", "city IS NOT NULL AND country IS NOT NULL")
def silver_franchises():
    """
    Enrich franchise data with supplier information.

    Transformations:
    - Join with supplier data for ingredient sourcing info
    - Categorize franchise by size
    """
    franchises = spark.read.table("LIVE.bronze_franchises")
    suppliers = spark.read.table("LIVE.bronze_suppliers")

    return (
        franchises
        .join(
            broadcast(suppliers.select(
                F.col("supplierID"),
                F.col("name").alias("supplier_name"),
                F.col("ingredient").alias("primary_ingredient"),
                F.col("approved").alias("supplier_approved"),
            )),
            on="supplierID",
            how="left"
        )
        # Categorize franchise size
        .withColumn(
            "size_category",
            F.when(F.col("size") == "Small", "S")
            .when(F.col("size") == "Medium", "M")
            .when(F.col("size") == "Large", "L")
            .otherwise("Unknown")
        )
    )


# =============================================================================
# GOLD LAYER
# =============================================================================
# Business-level aggregations ready for analytics and reporting.
# Design choice: Pre-computed aggregates for common analytical queries.
# =============================================================================


@dp.materialized_view(
    name="gold_daily_sales_summary",
    comment="Daily sales metrics aggregated by date and franchise",
    table_properties={"quality": "gold"},
    cluster_by=["transaction_date", "franchiseID"],
)
def gold_daily_sales_summary():
    """
    Daily sales summary by franchise.

    Metrics:
    - Total revenue and transactions
    - Average transaction value
    - Unique customers served
    - Product mix breakdown
    """
    return (
        spark.read.table("LIVE.silver_transactions")
        .groupBy("transaction_date", "franchiseID")
        .agg(
            F.sum("totalPrice").alias("total_revenue"),
            F.count("transactionID").alias("transaction_count"),
            F.avg("totalPrice").alias("avg_transaction_value"),
            F.countDistinct("customerID").alias("unique_customers"),
            F.sum("quantity").alias("total_items_sold"),
            F.countDistinct("product").alias("unique_products_sold"),
        )
    )


@dp.materialized_view(
    name="gold_customer_360",
    comment="Customer lifetime value and behavior metrics",
    table_properties={"quality": "gold"},
    cluster_by=["customerID"],
)
def gold_customer_360():
    """
    Customer 360 view with lifetime metrics.

    Metrics:
    - Lifetime value (total spend)
    - Purchase frequency
    - Favorite products and franchises
    - Recency metrics
    """
    transactions = spark.read.table("LIVE.silver_transactions")
    customers = spark.read.table("LIVE.silver_customers")

    customer_metrics = (
        transactions
        .groupBy("customerID")
        .agg(
            F.sum("totalPrice").alias("lifetime_value"),
            F.count("transactionID").alias("total_transactions"),
            F.avg("totalPrice").alias("avg_transaction_value"),
            F.min("transaction_date").alias("first_purchase_date"),
            F.max("transaction_date").alias("last_purchase_date"),
            F.countDistinct("franchiseID").alias("franchises_visited"),
            # Most purchased product (mode approximation)
            F.first("product").alias("sample_product"),
        )
    )

    return (
        customers
        .join(customer_metrics, on="customerID", how="left")
        .withColumn(
            "customer_tenure_days",
            F.datediff(F.col("last_purchase_date"), F.col("first_purchase_date"))
        )
        .withColumn(
            "purchase_frequency",
            F.when(F.col("customer_tenure_days") > 0,
                   F.col("total_transactions") / F.col("customer_tenure_days"))
            .otherwise(F.lit(0))
        )
        .select(
            "customerID",
            "full_name",
            "email_address",
            "region",
            "country_clean",
            "lifetime_value",
            "total_transactions",
            "avg_transaction_value",
            "first_purchase_date",
            "last_purchase_date",
            "customer_tenure_days",
            "purchase_frequency",
            "franchises_visited",
        )
    )


@dp.materialized_view(
    name="gold_product_performance",
    comment="Product-level sales performance metrics",
    table_properties={"quality": "gold"},
    cluster_by=["product"],
)
def gold_product_performance():
    """
    Product performance analysis.

    Metrics:
    - Total units sold and revenue
    - Average price and quantity per transaction
    - Franchise penetration (how many locations sell it)
    """
    return (
        spark.read.table("LIVE.silver_transactions")
        .groupBy("product")
        .agg(
            F.sum("quantity").alias("total_units_sold"),
            F.sum("totalPrice").alias("total_revenue"),
            F.count("transactionID").alias("transaction_count"),
            F.avg("unitPrice").alias("avg_unit_price"),
            F.avg("quantity").alias("avg_quantity_per_transaction"),
            F.countDistinct("franchiseID").alias("franchise_count"),
            F.countDistinct("customerID").alias("unique_customers"),
        )
        .withColumn(
            "revenue_per_unit",
            F.col("total_revenue") / F.col("total_units_sold")
        )
    )


@dp.materialized_view(
    name="gold_franchise_performance",
    comment="Franchise-level performance metrics with location details",
    table_properties={"quality": "gold"},
    cluster_by=["franchiseID"],
)
def gold_franchise_performance():
    """
    Franchise performance dashboard.

    Metrics:
    - Revenue and transaction volume
    - Customer metrics
    - Operational metrics (avg basket size, peak hours)
    """
    transactions = spark.read.table("LIVE.silver_transactions")
    franchises = spark.read.table("LIVE.silver_franchises")

    franchise_metrics = (
        transactions
        .groupBy("franchiseID")
        .agg(
            F.sum("totalPrice").alias("total_revenue"),
            F.count("transactionID").alias("total_transactions"),
            F.avg("totalPrice").alias("avg_transaction_value"),
            F.countDistinct("customerID").alias("unique_customers"),
            F.avg("quantity").alias("avg_items_per_transaction"),
            # Peak hour analysis (using expr for compatibility)
            F.expr("mode(transaction_hour)").alias("peak_hour"),
            # Weekend vs weekday split
            F.sum(F.when(F.col("is_weekend"), F.col("totalPrice")).otherwise(0)).alias("weekend_revenue"),
            F.sum(F.when(~F.col("is_weekend"), F.col("totalPrice")).otherwise(0)).alias("weekday_revenue"),
        )
    )

    return (
        broadcast(franchises)
        .join(franchise_metrics, on="franchiseID", how="left")
        .withColumn(
            "weekend_revenue_pct",
            F.round(F.col("weekend_revenue") / F.col("total_revenue") * 100, 2)
        )
        .select(
            "franchiseID",
            "name",
            "city",
            "country",
            "size_category",
            "supplier_name",
            "primary_ingredient",
            "total_revenue",
            "total_transactions",
            "avg_transaction_value",
            "unique_customers",
            "avg_items_per_transaction",
            "peak_hour",
            "weekend_revenue_pct",
            "latitude",
            "longitude",
        )
    )


@dp.materialized_view(
    name="gold_payment_analysis",
    comment="Payment method analysis and trends",
    table_properties={"quality": "gold"},
    cluster_by=["transaction_date", "payment_method_clean"],
)
def gold_payment_analysis():
    """
    Payment method breakdown.

    Metrics:
    - Revenue and volume by payment type
    - Average transaction value by payment method
    - Trend indicators
    """
    return (
        spark.read.table("LIVE.silver_transactions")
        .groupBy("payment_method_clean", "transaction_date")
        .agg(
            F.sum("totalPrice").alias("total_revenue"),
            F.count("transactionID").alias("transaction_count"),
            F.avg("totalPrice").alias("avg_transaction_value"),
        )
    )
