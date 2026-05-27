import io
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

# ── Dark-theme defaults so every generated chart matches the UI ──────────────
mpl.rcParams.update({
    "figure.facecolor":  "#0f172a",
    "axes.facecolor":    "#131c35",
    "axes.edgecolor":    "#1e3a5f",
    "axes.labelcolor":   "#cbd5e1",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "grid.color":        "#1e2d4a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "text.color":        "#e2e8f0",
    "legend.facecolor":  "#131c35",
    "legend.edgecolor":  "#1e3a5f",
    "legend.labelcolor": "#e2e8f0",
    "figure.dpi":        110,
})
# Seaborn palette that looks good on dark backgrounds
sns.set_theme(style="darkgrid", palette="muted")
sns.set_context("notebook")

_figures: list[bytes] = []


def get_figures() -> list[bytes]:
    return list(_figures)


def clear_figures() -> None:
    _figures.clear()


def execute_code(code: str, file_path: str | None = None) -> str:
    """Execute Python code in a sandboxed namespace, capturing stdout and matplotlib figures."""
    global _figures

    namespace: dict = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
    }

    if file_path:
        try:
            namespace["df"] = pd.read_csv(file_path)
        except Exception as exc:
            return f"Error loading CSV: {exc}"

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    plt.close("all")

    error: str | None = None
    try:
        exec(code, namespace)  # nosec — intentional code execution for demo
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    # Capture every matplotlib figure the code produced
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        _figures.append(buf.read())
    plt.close("all")

    output = buffer.getvalue()
    if error:
        return f"OUTPUT:\n{output}\n\nERROR:\n{error}"
    return output if output else "Code executed successfully (no printed output)."
