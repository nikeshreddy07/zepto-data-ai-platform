# Data Pipeline

## Requirements covered

- Scrapes at least three book categories from books.toscrape.com.
- Cleans price, star rating and availability.
- Uses the fixed project baseline `1 GBP = 105.50 INR`.
- Stores data in normalized SQLite tables `categories` and `books`.
- Demonstrates SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN and JOIN.
- Uses both `pd.read_sql` and `pd.merge`.

## Parsing decision

Rows with missing/unparseable required title, price or rating are dropped because they cannot form a valid product record. Availability is parsed into a boolean from the presence of `In stock`.

## Run

```bash
python scrape.py
python database.py
python queries.py
```
