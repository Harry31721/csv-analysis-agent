import pandas as pd
from langchain_core.tools import tool

from utils.code_executor import execute_code


@tool
def read_csv_file(file_path: str, n_rows: int = 10) -> str:
    """Read a CSV file and return its first rows, column names, and data types.

    Args:
        file_path: Absolute path to the CSV file.
        n_rows: Number of rows to preview (default 10).
    """
    try:
        df = pd.read_csv(file_path)
        result = (
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            f"Columns: {list(df.columns)}\n\n"
            f"Data types:\n{df.dtypes.to_string()}\n\n"
            f"First {n_rows} rows:\n{df.head(n_rows).to_string()}"
        )
        return result
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool
def analyze_dataframe(file_path: str) -> str:
    """Get a comprehensive statistical profile of a CSV file.

    Reports shape, data types, missing values, duplicate rows, descriptive
    statistics for numeric columns, and value counts for low-cardinality
    categorical columns.

    Args:
        file_path: Absolute path to the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        lines: list[str] = []

        lines.append("=== DATA PROFILE ===")
        lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

        lines.append(f"\nColumn types:\n{df.dtypes.to_string()}")

        nulls = df.isnull().sum()
        if nulls.any():
            lines.append(f"\nMissing values:\n{nulls[nulls > 0].to_string()}")
        else:
            lines.append("\nMissing values: none")

        dup_count = df.duplicated().sum()
        lines.append(f"\nDuplicate rows: {dup_count}")
        if dup_count:
            lines.append(f"Duplicate row indices: {list(df[df.duplicated(keep=False)].index[:10])}")

        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            lines.append(f"\nNumeric statistics:\n{numeric.describe().to_string()}")

        for col in df.select_dtypes(include="object").columns:
            n_unique = df[col].nunique()
            if n_unique <= 12:
                lines.append(f"\n'{col}' value counts (unique={n_unique}):\n{df[col].value_counts(dropna=False).to_string()}")

        return "\n".join(lines)
    except Exception as exc:
        return f"Error profiling file: {exc}"


@tool
def execute_python_code(code: str, file_path: str = "") -> str:
    """Execute Python code for data analysis and visualization.

    A pandas DataFrame named 'df' is pre-loaded from file_path when provided.
    Available libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns).

    Rules:
    - Use print() for any output you want displayed.
    - Create matplotlib/seaborn figures normally — they are captured automatically.
    - Do NOT call plt.show() — it has no effect.
    - Return values are not shown; use print() instead.

    Args:
        code: Python code to execute.
        file_path: Path to the CSV file so df is pre-loaded (recommended).
    """
    return execute_code(code, file_path if file_path else None)
