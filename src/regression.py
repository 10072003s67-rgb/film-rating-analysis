"""Reproduce the movie-rating regression and save a diagnostic chart."""

from pathlib import Path
import argparse
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["budget", "runtime", "popularity", "vote_count", "release_year"]


def main(database: Path, output: Path) -> None:
    with sqlite3.connect(database) as connection:
        df = pd.read_sql_query(
            "SELECT rating, budget, runtime, popularity, vote_count, "
            "release_year, genre_1 FROM dataset_5000", connection
        )
    for column in ["rating", *NUMERIC]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["genre_1"] = df["genre_1"].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=["rating", *NUMERIC, "genre_1"])
    df = df.loc[(df["rating"] > 0) & (df["budget"] > 0)
                & (df["runtime"] > 0) & (df["release_year"] > 0)].copy()

    features = [*NUMERIC, "genre_1"]
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df["rating"], test_size=0.2, random_state=33
    )
    model = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("genre", OneHotEncoder(handle_unknown="ignore"), ["genre_1"])
        ], remainder="passthrough")),
        ("regression", LinearRegression()),
    ])
    model.fit(x_train, y_train)

    print(f"Films: {len(df)} | train: {len(x_train)} | test: {len(x_test)}")
    for label, x, y in [("Train", x_train, y_train), ("Test", x_test, y_test)]:
        prediction = model.predict(x)
        print(f"{label}: R²={r2_score(y, prediction):.4f}, "
              f"MAE={mean_absolute_error(y, prediction):.4f}, "
              f"RMSE={np.sqrt(mean_squared_error(y, prediction)):.4f}")

    coefs = model.named_steps["regression"].coef_[-len(NUMERIC):]
    print("Numeric coefficients:")
    for name, value in zip(NUMERIC, coefs):
        print(f"  {name}: {value:.10f}")

    prediction = model.predict(x_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, prediction, s=15, alpha=0.45, color="#287b8d")
    bounds = [min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())]
    ax.plot(bounds, bounds, linestyle="--", color="#c0583b", label="Идеальный прогноз")
    ax.set(xlabel="Фактический рейтинг", ylabel="Прогноз модели",
           title="Фактические и прогнозируемые рейтинги")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Chart: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "DS.db")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "actual_vs_predicted.png")
    args = parser.parse_args()
    main(args.database, args.output)
