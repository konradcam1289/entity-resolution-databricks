# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze ingest
# MAGIC
# MAGIC Loads the raw Abt-Buy CSV files from the Unity Catalog volume into Delta tables in the **bronze** layer.
# MAGIC Data is stored as-is (all columns as strings) plus ingestion metadata; cleaning happens in the silver layer.
# MAGIC
# MAGIC | Table | Source file | Content |
# MAGIC |---|---|---|
# MAGIC | `bronze.abt` | `Abt.csv` | products from the Abt shop |
# MAGIC | `bronze.buy` | `Buy.csv` | products from the Buy shop |
# MAGIC | `bronze.perfect_mapping` | `abt_buy_perfectMapping.csv` | ground-truth matching pairs |

# COMMAND ----------

CATALOG = "workspace"
RAW_PATH = f"/Volumes/{CATALOG}/raw/files"
BRONZE = f"{CATALOG}.bronze"

# file name, encoding (Abt.csv is Latin-1, not UTF-8), expected row count, table description
SOURCES = {
    "abt": ("Abt.csv", "ISO-8859-1", 1081, "Raw products from the Abt shop (Abt-Buy benchmark)"),
    "buy": ("Buy.csv", "UTF-8", 1092, "Raw products from the Buy shop (Abt-Buy benchmark)"),
    "perfect_mapping": ("abt_buy_perfectMapping.csv", "UTF-8", 1097, "Ground-truth matching pairs between Abt and Buy"),
}

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {BRONZE} COMMENT 'Bronze layer: raw source data loaded as-is'")

# COMMAND ----------

from pyspark.sql import functions as F


def read_raw_csv(file_name: str, encoding: str):
    """Read a CSV file without type inference, so the bronze layer keeps the source values unchanged."""
    return (
        spark.read
        .option("header", True)
        .option("multiLine", True)  # descriptions may contain line breaks inside quotes
        .option("escape", '"')
        .option("encoding", encoding)
        .csv(f"{RAW_PATH}/{file_name}")
        .withColumn("_source_file", F.lit(file_name))
        .withColumn("_ingested_at", F.current_timestamp())
    )

# COMMAND ----------

for table, (file_name, encoding, expected_rows, description) in SOURCES.items():
    full_name = f"{BRONZE}.{table}"
    df = read_raw_csv(file_name, encoding)

    row_count = df.count()
    assert row_count == expected_rows, f"{full_name}: expected {expected_rows} rows, got {row_count}"

    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_name)
    spark.sql(f"COMMENT ON TABLE {full_name} IS '{description}'")
    print(f"{full_name}: {row_count} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick look at the data

# COMMAND ----------

display(spark.table(f"{BRONZE}.abt").limit(5))

# COMMAND ----------

display(spark.table(f"{BRONZE}.buy").limit(5))

# COMMAND ----------

display(spark.table(f"{BRONZE}.perfect_mapping").limit(5))
