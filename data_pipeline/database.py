import sqlite3
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent
CSV = HERE / "books_clean.csv"
DB = HERE / "zepto_books.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY(category_id) REFERENCES categories(category_id)
);
"""

def build_database():
    df = pd.read_csv(CSV)

    with sqlite3.connect(DB) as con:
        con.executescript("DROP TABLE IF EXISTS books; DROP TABLE IF EXISTS categories;")
        con.executescript(SCHEMA)

        cats = pd.DataFrame({"category_name": sorted(df["category"].dropna().unique())})
        cats.to_sql("categories", con, if_exists="append", index=False)

        mapping = pd.read_sql("SELECT category_id, category_name FROM categories", con)
        out = df.merge(mapping, left_on="category", right_on="category_name")
        books = out[[
            "title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"
        ]].copy()
        books["in_stock"] = books["in_stock"].astype(int)
        books.to_sql("books", con, if_exists="append", index=False)

    print(f"Database created: {DB}")

if __name__ == "__main__":
    build_database()
