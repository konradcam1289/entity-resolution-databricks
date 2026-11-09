# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver clean
# MAGIC
# MAGIC Cleans and standardises the bronze tables so that records from both shops can be compared.
# MAGIC
# MAGIC | Table | Content |
# MAGIC |---|---|
# MAGIC | `silver.abt` | Abt products in a common product schema |
# MAGIC | `silver.buy` | Buy products in the same schema |
# MAGIC | `silver.perfect_mapping` | ground-truth pairs with typed ids, checked against both product tables |
# MAGIC | `silver.dq_check_results` | log of every data quality check run (append-only) |
# MAGIC
# MAGIC **Key decisions**
# MAGIC - Cleaning logic lives in `src/er_pipeline/cleaning.py` as native Spark column expressions (no Python UDFs),
# MAGIC   so it is fast, reusable in later notebooks and unit-testable.
# MAGIC - Model numbers are normalised to letters and digits only (`PS-LX350H` → `pslx350h`), because the shops format them differently.
# MAGIC - Prices are parsed to `DECIMAL(12,2)`, not `DOUBLE`, because they are money.
# MAGIC - **Data contract**: every table is created from an explicit schema (types, `NOT NULL`, column comments, `CHECK` constraints).
# MAGIC   Data quality checks run *before* anything is written, so a failing check leaves the previous silver tables untouched.

# COMMAND ----------

import os
import sys

# The notebook runs from <repo>/notebooks, so <repo>/src holds the importable package.
SRC_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from pyspark.sql import functions as F

from er_pipeline import cleaning
from er_pipeline.quality import (
    WARN,
    Check,
    duplicate_key_check,
    orphan_check,
    raise_on_errors,
    run_row_checks,
)

CATALOG = "workspace"
BRONZE = f"{CATALOG}.bronze"
SILVER = f"{CATALOG}.silver"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER} COMMENT 'Silver layer: cleaned, typed and validated data'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform
# MAGIC Both shops are mapped to one common product schema. Abt has no `manufacturer` column,
# MAGIC so the brand is taken from the first token of the name (product names start with the brand in both shops).

# COMMAND ----------


def build_products(bronze_df, source: str):
    manufacturer = F.col("manufacturer") if "manufacturer" in bronze_df.columns else F.lit(None).cast("string")
    name_clean = cleaning.normalize_text(F.col("name"))
    name_tokens = cleaning.tokenize(name_clean)

    return bronze_df.select(
        F.expr("try_cast(id AS BIGINT)").alias("id"),
        F.lit(source).alias("source"),
        F.trim("name").alias("name"),
        name_clean.alias("name_clean"),
        name_tokens.alias("name_tokens"),
        cleaning.model_number(F.col("name")).alias("model_number"),
        cleaning.model_tokens(F.col("name")).alias("model_tokens"),
        F.coalesce(
            cleaning.normalize_text(manufacturer),
            F.try_element_at(name_tokens, F.lit(1)),
        ).alias("brand"),
        cleaning.normalize_text(F.col("description")).alias("description_clean"),
        cleaning.parse_price(F.col("price")).alias("price"),
        cleaning.blank_to_null(F.col("price")).alias("price_raw"),
        "_source_file",
        "_ingested_at",
        F.current_timestamp().alias("_processed_at"),
    )


abt = build_products(spark.table(f"{BRONZE}.abt"), "abt")
buy = build_products(spark.table(f"{BRONZE}.buy"), "buy")

mapping = spark.table(f"{BRONZE}.perfect_mapping").select(
    F.expr("try_cast(idAbt AS BIGINT)").alias("id_abt"),
    F.expr("try_cast(idBuy AS BIGINT)").alias("id_buy"),
    "_source_file",
    "_ingested_at",
    F.current_timestamp().alias("_processed_at"),
)

display(abt.select("name", "name_tokens", "model_number", "model_tokens", "brand", "price_raw", "price").limit(10))

# COMMAND ----------

display(buy.select("name", "name_tokens", "model_number", "model_tokens", "brand", "price_raw", "price").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling
# MAGIC How complete are the derived attributes? This tells us which ones are reliable enough for blocking and matching.

# COMMAND ----------


def pct(condition):
    """Share of rows (in %) where the condition is true; NULL counts as false."""
    return F.round(100 * F.avg(F.coalesce(condition, F.lit(False)).cast("int")), 1)


display(
    abt.unionByName(buy)
    .groupBy("source")
    .agg(
        F.count("*").alias("rows"),
        pct(F.col("model_number").isNotNull()).alias("pct_model_number"),
        pct(F.size("model_tokens") > 0).alias("pct_any_model_token"),
        pct(F.col("price").isNotNull()).alias("pct_price"),
        pct(F.col("description_clean").isNotNull()).alias("pct_description"),
        F.round(F.avg(F.size("name_tokens")), 1).alias("avg_name_tokens"),
    )
    .orderBy("source")
)

# COMMAND ----------

# MAGIC %md
# MAGIC For the ground-truth pairs: how often do both records share a model number?
# MAGIC A high share means model numbers are a strong signal for blocking (next notebook).

# COMMAND ----------

true_pairs = (
    mapping.select("id_abt", "id_buy")
    .join(abt.select(F.col("id").alias("id_abt"), F.col("model_number").alias("abt_model"),
                     F.col("model_tokens").alias("abt_codes")), "id_abt")
    .join(buy.select(F.col("id").alias("id_buy"), F.col("model_tokens").alias("buy_codes")), "id_buy")
)

display(
    true_pairs.agg(
        F.count("*").alias("true_pairs"),
        pct(F.array_contains("buy_codes", F.col("abt_model"))).alias("pct_abt_model_in_buy_name"),
        pct(F.arrays_overlap("abt_codes", "buy_codes")).alias("pct_any_shared_code"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data quality checks (data contract)
# MAGIC `error` checks stop the pipeline, `warn` checks are only reported. Every result is appended to `silver.dq_check_results`.

# COMMAND ----------


def product_checks():
    return [
        Check("id_not_null", F.col("id").isNull()),
        Check("name_clean_not_null", F.col("name_clean").isNull()),
        Check("price_parsed", F.col("price_raw").isNotNull() & F.col("price").isNull()),
        Check("price_non_negative", F.col("price") < 0),
        Check("model_number_present", F.col("model_number").isNull(), WARN),
        Check("model_tokens_present", F.size("model_tokens") == 0, WARN),
        Check("price_present", F.col("price").isNull(), WARN),
    ]


results = [
    *run_row_checks(abt, "silver.abt", product_checks()),
    duplicate_key_check(abt, "silver.abt", ["id"]),
    *run_row_checks(buy, "silver.buy", product_checks()),
    duplicate_key_check(buy, "silver.buy", ["id"]),
    *run_row_checks(mapping, "silver.perfect_mapping", [
        Check("id_abt_not_null", F.col("id_abt").isNull()),
        Check("id_buy_not_null", F.col("id_buy").isNull()),
    ]),
    duplicate_key_check(mapping, "silver.perfect_mapping", ["id_abt", "id_buy"]),
    orphan_check(mapping, "id_abt", abt, "id", "silver.perfect_mapping"),
    orphan_check(mapping, "id_buy", buy, "id", "silver.perfect_mapping"),
]

results_df = spark.createDataFrame(
    [(r.table, r.check, r.severity, r.failed_rows, r.passed) for r in results],
    "table_name STRING, check_name STRING, severity STRING, failed_rows BIGINT, passed BOOLEAN",
).withColumn("run_at", F.current_timestamp())

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {SILVER}.dq_check_results (
        table_name STRING, check_name STRING, severity STRING,
        failed_rows BIGINT, passed BOOLEAN, run_at TIMESTAMP
    ) COMMENT 'Append-only log of data quality check results for the silver layer'
""")
results_df.write.mode("append").saveAsTable(f"{SILVER}.dq_check_results")

display(results_df)
raise_on_errors(results)  # nothing below runs if an "error" check failed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write silver tables
# MAGIC Each table is (re)created from its contract: column types, `NOT NULL`, column comments and `CHECK` constraints.
# MAGIC The data is then appended into the empty table, so Delta enforces the schema and constraints on write
# MAGIC (a wrong type or a NULL id fails the write instead of silently changing the table).

# COMMAND ----------

LINEAGE_COLUMNS = [
    ("_source_file", "STRING", "Source file name in the raw volume"),
    ("_ingested_at", "TIMESTAMP", "When the record was loaded into bronze"),
    ("_processed_at", "TIMESTAMP", "When the record was written to silver"),
]

PRODUCT_CONTRACT = [
    ("id", "BIGINT NOT NULL", "Product id in the source shop"),
    ("source", "STRING NOT NULL", "Source shop: abt or buy"),
    ("name", "STRING NOT NULL", "Original product name (trimmed)"),
    ("name_clean", "STRING NOT NULL", "Name in lower case with special characters replaced by spaces"),
    ("name_tokens", "ARRAY<STRING>", "Distinct name tokens without stopwords"),
    ("model_number", "STRING", "Normalised model number from the name suffix after a dash, if present"),
    ("model_tokens", "ARRAY<STRING>", "Normalised model number candidates found in the name"),
    ("brand", "STRING", "Manufacturer (Buy) or first name token, normalised"),
    ("description_clean", "STRING", "Normalised product description"),
    ("price", "DECIMAL(12,2)", "Price in USD parsed from price_raw"),
    ("price_raw", "STRING", "Original price text, e.g. $399.00"),
    *LINEAGE_COLUMNS,
]

MAPPING_CONTRACT = [
    ("id_abt", "BIGINT NOT NULL", "Product id in silver.abt"),
    ("id_buy", "BIGINT NOT NULL", "Product id in silver.buy"),
    *LINEAGE_COLUMNS,
]

PRODUCT_CONSTRAINTS = {
    "price_non_negative": "price IS NULL OR price >= 0",
    "source_valid": "source IN ('abt', 'buy')",
}


def write_with_contract(df, table: str, contract, comment: str, constraints=None):
    full_name = f"{SILVER}.{table}"
    columns_ddl = ",\n  ".join(f"{name} {col_type} COMMENT '{col_comment}'" for name, col_type, col_comment in contract)
    spark.sql(f"CREATE OR REPLACE TABLE {full_name} (\n  {columns_ddl}\n) COMMENT '{comment}'")
    for name, expression in (constraints or {}).items():
        spark.sql(f"ALTER TABLE {full_name} DROP CONSTRAINT IF EXISTS {name}")  # safe re-runs
        spark.sql(f"ALTER TABLE {full_name} ADD CONSTRAINT {name} CHECK ({expression})")

    typed = df.select([F.col(name).cast(col_type.replace(" NOT NULL", "")).alias(name) for name, col_type, _ in contract])
    typed.write.mode("append").saveAsTable(full_name)
    print(f"{full_name}: {spark.table(full_name).count()} rows")


write_with_contract(abt, "abt", PRODUCT_CONTRACT, "Cleaned Abt products", PRODUCT_CONSTRAINTS)
write_with_contract(buy, "buy", PRODUCT_CONTRACT, "Cleaned Buy products", PRODUCT_CONSTRAINTS)
write_with_contract(mapping, "perfect_mapping", MAPPING_CONTRACT, "Validated ground-truth matching pairs between Abt and Buy")

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE EXTENDED workspace.silver.abt
