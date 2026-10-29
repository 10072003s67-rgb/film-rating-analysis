"""Rank directors with at least five films in the dataset."""

from pathlib import Path
import argparse
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main(database: Path, output: Path) -> None:
    with sqlite3.connect(database) as connection:
        df = pd.read_sql_query(
            "SELECT movie_id, rating, director_1, director_2 FROM dataset_5000",
            connection,
        )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    directors = df.melt(id_vars=["movie_id", "rating"],
                        value_vars=["director_1", "director_2"], value_name="director")
    directors["director"] = directors["director"].fillna("").str.strip()
    directors = directors.loc[directors["director"] != ""]
    directors = directors.drop_duplicates(["movie_id", "director"])
    stats = directors.groupby("director").agg(
        mean_rating=("rating", "mean"), film_count=("movie_id", "nunique")
    ).query("film_count >= 5")
    top = stats.sort_values(["mean_rating", "film_count"], ascending=False).head(15)
    top = top.sort_values("mean_rating")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(top.index, top["mean_rating"], color="#287b8d")
    for bar, row in zip(bars, top.itertuples()):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{row.mean_rating:.2f}  (n={row.film_count})", va="center", fontsize=9)
    ax.set(xlim=(0, 10), xlabel="Средний рейтинг",
           title="Режиссёры с не менее чем 5 фильмами")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(top.sort_values("mean_rating", ascending=False).to_string())
    print(f"Chart: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "DS.db")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "top_directors.png")
    args = parser.parse_args()
    main(args.database, args.output)
