import sqlite3
from pathlib import Path
import pandas as pd

DB = Path(__file__).parent / "zepto_books.db"

QUERIES = {
    "select_where": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4;
    """,
    "order_by": """
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC;
    """,
    "limit": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    "distinct": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,
    "between": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30;
    """,
    "join": """
        SELECT c.category_name, b.title, b.rating, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, c.category_name, b.title;
    """
}

def main():
    with sqlite3.connect(DB) as con:
        for name, query in QUERIES.items():
            print("\n" + "=" * 70)
            print(name.upper())
            print(query.strip())
            result = pd.read_sql(query, con)
            print(result.to_string(index=False))

        join_sql = pd.read_sql(QUERIES["join"], con)
        books = pd.read_sql("SELECT * FROM books", con)
        categories = pd.read_sql("SELECT * FROM categories", con)

    join_merge = books.merge(
        categories, on="category_id", how="inner"
    )[["category_name", "title", "rating", "price_inr"]].sort_values(
        ["rating", "category_name", "title"], ascending=[False, True, True]
    ).reset_index(drop=True)

    print("\nJOIN via pd.merge:")
    print(join_merge.to_string(index=False))
    print("\nEquivalent:", join_sql.reset_index(drop=True).equals(join_merge))

if __name__ == "__main__":
    main()
