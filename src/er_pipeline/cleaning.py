"""Column-level cleaning functions for product records.

Every function takes and returns a PySpark ``Column`` expression, so the logic runs natively
inside Spark (no Python UDFs, no row-by-row serialisation) and can be unit-tested with a local
SparkSession.
"""

from pyspark.sql import Column
from pyspark.sql import functions as F

# Very common words that carry no information for matching product names.
STOPWORDS = ["a", "an", "and", "for", "in", "of", "on", "the", "to", "with"]

# A product-code-like chunk of raw text: alphanumerics optionally joined by - / .
# e.g. "KDL-52XBR4", "PS-LX350H", "EZXS88W".
CODE_PATTERN = r"[A-Za-z0-9]+(?:[-/.][A-Za-z0-9]+)*"

# Abt names usually end with " - <MODEL>", e.g. "Sony Turntable - PSLX350H".
MODEL_SUFFIX_PATTERN = r"\s-\s*([A-Za-z0-9][A-Za-z0-9./-]*)\s*$"

# Normalised tokens that contain digits and letters but are measurements, not model numbers
# (e.g. "1080p", "24ghz" from "2.4GHz", "8mp", "51ch" from "5.1ch", "16x").
UNIT_PATTERN = (
    r"^[0-9]+(?:mm|cm|in|ft|gb|mb|tb|kb|mp|hz|khz|mhz|ghz|w|v|mah|lb|lbs|oz|p|x|ch|st|nd|rd|th)$"
)


def blank_to_null(col: Column) -> Column:
    """Trim a string and turn empty strings into NULL."""
    trimmed = F.trim(col)
    return F.when(trimmed != "", trimmed)


def normalize_text(col: Column) -> Column:
    """Lower-case and replace every run of non-alphanumeric characters with one space.

    "Sony® Bravia 52'  LCD-TV" -> "sony bravia 52 lcd tv". Empty results become NULL.
    """
    cleaned = F.trim(F.regexp_replace(F.lower(col), r"[^a-z0-9]+", " "))
    return F.when(cleaned != "", cleaned)


def tokenize(clean_col: Column) -> Column:
    """Split normalised text into a set of tokens (distinct, stopwords removed)."""
    tokens = F.filter(F.split(clean_col, " "), lambda t: (t != "") & ~t.isin(STOPWORDS))
    return F.array_distinct(tokens)


def normalize_code(col: Column) -> Column:
    """Canonical form of a model number: lower case, letters and digits only.

    Shops write the same model differently ("PS-LX350H" vs "PSLX350H"), so separators are dropped.
    """
    return F.regexp_replace(F.lower(col), r"[^a-z0-9]", "")


def model_number(raw_name: Column) -> Column:
    """Model number taken from the " - <MODEL>" suffix of a product name, or NULL.

    The suffix must contain a digit and have at least 3 characters, so endings such as
    " - Black" are not treated as model numbers.
    """
    code = normalize_code(F.regexp_extract(raw_name, MODEL_SUFFIX_PATTERN, 1))
    return F.when(code.rlike("[0-9]") & (F.length(code) >= 3), code)


def model_tokens(raw_name: Column) -> Column:
    """All model-number candidates found anywhere in a product name (normalised, distinct).

    A candidate has at least 4 characters, mixes letters and digits and is not a unit of measure.
    The suffix model number (see ``model_number``) is always included.
    """
    codes = F.transform(F.regexp_extract_all(raw_name, F.lit(CODE_PATTERN), 0), normalize_code)
    codes = F.filter(
        codes,
        lambda c: (F.length(c) >= 4) & c.rlike("[a-z]") & c.rlike("[0-9]") & ~c.rlike(UNIT_PATTERN),
    )
    return F.array_distinct(F.array_compact(F.concat(codes, F.array(model_number(raw_name)))))


def parse_price(col: Column) -> Column:
    """Parse a price string such as "$1,299.00" into DECIMAL(12,2); empty or missing -> NULL.

    DECIMAL instead of DOUBLE because money must not suffer from binary floating point rounding.
    The regex guarantees a valid number, so the cast cannot fail even with ANSI mode enabled.
    """
    number = F.regexp_replace(F.regexp_extract(col, r"([0-9][0-9,]*(?:\.[0-9]+)?)", 1), ",", "")
    return F.when(number != "", number).cast("decimal(12,2)")
