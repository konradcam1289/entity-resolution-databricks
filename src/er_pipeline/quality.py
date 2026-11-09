"""Lightweight data quality checks (a simple data contract) for Spark DataFrames.

A check counts the rows that violate a rule. Checks with severity "error" stop the pipeline,
checks with severity "warn" are only reported. All results can be logged to a Delta table.
"""

from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

ERROR = "error"
WARN = "warn"


@dataclass(frozen=True)
class Check:
    name: str
    is_violated: Column  # boolean expression, TRUE for rows that break the rule
    severity: str = ERROR


@dataclass(frozen=True)
class CheckResult:
    table: str
    check: str
    severity: str
    failed_rows: int

    @property
    def passed(self) -> bool:
        return self.failed_rows == 0


def run_row_checks(df: DataFrame, table: str, checks: list[Check]) -> list[CheckResult]:
    """Evaluate all row-level checks in a single aggregation (one pass over the data)."""
    counts = df.agg(*[F.count_if(c.is_violated).alias(c.name) for c in checks]).first()
    return [CheckResult(table, c.name, c.severity, int(counts[c.name])) for c in checks]


def duplicate_key_check(df: DataFrame, table: str, key_cols: list[str], severity: str = ERROR) -> CheckResult:
    """Number of rows whose key appears more than once."""
    duplicated = (
        df.groupBy(*key_cols).count()
        .filter(F.col("count") > 1)
        .agg(F.coalesce(F.sum("count"), F.lit(0)).alias("rows"))
        .first()["rows"]
    )
    return CheckResult(table, f"unique_{'_'.join(key_cols)}", severity, int(duplicated))


def orphan_check(
    child: DataFrame, child_col: str, parent: DataFrame, parent_col: str, table: str, severity: str = ERROR
) -> CheckResult:
    """Referential integrity: rows of ``child`` whose key does not exist in ``parent``."""
    parent_keys = parent.select(F.col(parent_col).alias("_parent_key")).distinct()
    orphans = child.join(parent_keys, F.col(child_col) == F.col("_parent_key"), "left_anti").count()
    return CheckResult(table, f"{child_col}_exists_in_parent", severity, orphans)


def raise_on_errors(results: list[CheckResult]) -> None:
    """Fail the pipeline if any "error" check did not pass."""
    failed = [r for r in results if r.severity == ERROR and not r.passed]
    if failed:
        details = ", ".join(f"{r.table}.{r.check} ({r.failed_rows} rows)" for r in failed)
        raise AssertionError(f"Data quality checks failed: {details}")
