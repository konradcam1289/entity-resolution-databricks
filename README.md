# Entity Resolution Pipeline on Databricks

PySpark pipeline that matches product records describing the same item across two online shops,
built on Databricks with Delta Lake and Unity Catalog in a Medallion architecture (bronze → silver → gold).

> 🚧 Work in progress – this README will be extended with the architecture diagram, design decisions and results.

## Pipeline

| Layer | Notebook | What it does |
|---|---|---|
| Bronze | `notebooks/01_bronze_ingest.py` | Loads raw CSV files from a Unity Catalog volume into Delta tables |
| Silver | *planned* | Cleaning and standardisation of product names, model numbers and prices |
| Gold | *planned* | Blocking, similarity features, Spark MLlib classifier, precision / recall / F1 |

## Dataset

[Abt-Buy](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution) benchmark
from the Database Group, Leipzig University (CC BY 4.0):

- `Abt.csv` – 1,081 products (id, name, description, price)
- `Buy.csv` – 1,092 products (id, name, description, manufacturer, price)
- `abt_buy_perfectMapping.csv` – 1,097 ground-truth matching pairs

Source: H. Köpcke, A. Thor, E. Rahm, *Evaluation of entity resolution approaches on real-world match problems*, PVLDB 3(1), 2010.

## How to run

1. Upload the three CSV files to the Unity Catalog volume `/Volumes/workspace/raw/files`.
2. Clone this repository as a Git folder in Databricks.
3. Run the notebooks in order on serverless compute.
